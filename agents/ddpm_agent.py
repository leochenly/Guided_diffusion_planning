# d3il/agents/ddpm_agent.py
from collections import deque
import os
import logging
from typing import Optional, Literal, Dict, Any

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import einops
from sklearn.neighbors import KernelDensity

from agents.base_agent import BaseAgent
from agents.models.diffusion.ema import ExponentialMovingAverage

log = logging.getLogger(__name__)


class DiffusionAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            use_ema: bool,
            discount: int,
            decay: float,
            update_ema_every_n_steps: int,
            goal_window_size: int,
            window_size: int,
            pred_last_action_only: bool = False,
            diffusion_kde: bool = False,
            diffusion_kde_samples: int = 100,
            goal_conditioned: bool = False,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # bounds for sampler
        self.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.eval_model_name = "eval_best_ddpm.pth"
        self.last_model_name = "last_ddpm.pth"

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )

        self.steps = 0

        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), decay, self.device)
        self.use_ema = use_ema
        self.discount = discount
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps

        self.goal_window_size = goal_window_size
        self.window_size = window_size
        self.pred_last_action_only = pred_last_action_only

        self.goal_condition = goal_conditioned

        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.goal_window_size)

        if not self.pred_last_action_only:
            self.action_context = deque(maxlen=self.window_size - 1)
            self.que_actions = True
        else:
            self.que_actions = False

        self.diffusion_kde = diffusion_kde
        self.diffusion_kde_samples = diffusion_kde_samples

    def train_agent(self):

        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    if self.goal_condition:
                        state, action, mask, goal = data
                        mean_mse = self.evaluate(state, action, goal)
                    else:
                        state, action, mask = data
                        mean_mse = self.evaluate(state, action)

                    test_mse.append(mean_mse)

                avrg_test_mse = sum(test_mse) / len(test_mse)
                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
                    wandb.log({"best_model_epochs": num_epoch})
                    log.info('New best test loss. Stored weights have been updated!')

            train_loss = []
            for data in self.train_dataloader:
                if self.goal_condition:
                    state, action, mask, goal = data
                    batch_loss = self.train_step(state, action, goal)
                else:
                    state, action, mask = data
                    batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)
                wandb.log({"loss": batch_loss})

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:
        self.model.train()
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        loss = self.model.loss(action, state, goal)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss

    @torch.no_grad()
    def evaluate(self, state: torch.tensor, action: torch.tensor, goal: Optional[torch.Tensor] = None) -> float:
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()
        loss = self.model.loss(action, state, goal)
        total_mse = loss.mean().item()

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return total_mse

    def reset(self):
        """Resets the context of the model."""
        self.obs_context.clear()

    def _stack_obs_context(self, state_scaled: torch.Tensor, pad_to_window: bool) -> torch.Tensor:
        """
        state_scaled: (1, obs_dim)
        returns input_state: (1, T, obs_dim) where T==len(context) or window_size if padded
        """
        self.obs_context.append(state_scaled)

        ctx = list(self.obs_context)  # each: (1, obs_dim)
        if pad_to_window and self.window_size > 1 and len(ctx) < self.window_size:
            # left-pad with the earliest available obs so that the latest obs still sits at the end
            pad = [ctx[0]] * (self.window_size - len(ctx))
            ctx = pad + ctx

        input_state = torch.stack(tuple(ctx), dim=1)  # (1, T, obs_dim)
        return input_state

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,
        goal: Optional[np.ndarray] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        *,
        return_sequence: bool = False,
        pad_to_window: bool = False,
        action_index: Literal["last", "first"] = "last",
    ) -> np.ndarray:
        """
        Default (backward-compatible): returns a single action (1, action_dim) as numpy.
        If return_sequence=True: returns the whole predicted action sequence (T, action_dim).
        pad_to_window=True: pad context to window_size before feeding transformer (recommended for trajectory mode).
        action_index selects which step to return when return_sequence=False.
        """
        if extra_args is None:
            extra_args = {}

        # (1, obs_dim)
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        state_t = self.scaler.scale_input(state_t)

        goal_t = None
        if goal is not None:
            goal_t = torch.from_numpy(goal).float().to(self.device)
            goal_t = self.scaler.scale_input(goal_t)

        # build transformer window input
        if self.window_size > 1:
            input_state = self._stack_obs_context(state_t, pad_to_window=pad_to_window)
            if goal_t is not None:
                goal_t = einops.rearrange(goal_t, 'd -> 1 1 d')
        else:
            input_state = state_t

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()

        # ----- inference -----
        if self.diffusion_kde:
            # NOTE: KDE selection is defined over a single vector. If return_sequence=True,
            # we still select using the chosen action_index slice.
            state_rpt = torch.repeat_interleave(input_state, repeats=self.diffusion_kde_samples, dim=0)
            goal_rpt = None
            if goal_t is not None:
                goal_rpt = torch.repeat_interleave(goal_t, repeats=self.diffusion_kde_samples, dim=0)

            x_0 = self.model(state_rpt, goal_rpt, **extra_args)  # (K,T,D) or (K,D)

            if x_0.dim() == 3:
                x_sel = x_0[:, -1, :] if action_index == "last" else x_0[:, 0, :]
            else:
                x_sel = x_0

            x_kde = x_sel.detach().cpu()
            kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(x_kde)
            kde_prob = kde.score_samples(x_kde)
            max_index = int(kde_prob.argmax(axis=0))

            model_pred = x_0[max_index:max_index+1]  # keep batch dim (1,...) for scaling
        else:
            model_pred = self.model(input_state, goal_t, **extra_args)  # (1,T,D) or (1,D)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        # inverse scale outputs (works for 2D or 3D tensors)
        model_pred = self.scaler.inverse_scale_output(model_pred)

        # ----- return formatting -----
        if return_sequence:
            # want (T,D) numpy
            if model_pred.dim() == 2:
                out = model_pred[0].unsqueeze(0)  # (1,D) -> (1,D) as T=1
            else:
                out = model_pred[0]               # (1,T,D) -> (T,D)
            return out.detach().cpu().numpy()

        # backward compatible: return one action (1,D)
        if model_pred.dim() == 3 and model_pred.size(1) > 1:
            model_pred = model_pred[:, -1, :] if action_index == "last" else model_pred[:, 0, :]
        elif model_pred.dim() == 3 and model_pred.size(1) == 1:
            model_pred = model_pred[:, 0, :]

        return model_pred.detach().cpu().numpy()

    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), self.decay, self.device)
        log.info('Loaded pre-trained model parameters')

