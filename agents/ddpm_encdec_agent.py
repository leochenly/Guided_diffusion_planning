from collections import deque
import os
import logging
from typing import Optional

from omegaconf import DictConfig
import hydra
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import einops
from sklearn.neighbors import KernelDensity

from agents.base_agent import BaseAgent
from agents.models.diffusion.ema import ExponentialMovingAverage

# A logger for this file
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
            obs_seq_len: int,
            action_seq_size: int,
            pred_last_action_only: bool = False,
            diffusion_kde: bool = False,
            diffusion_kde_samples: int = 100,
            goal_conditioned: bool = False,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # Define the bounds for the sampler class
        self.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.optimization = optimization
        self.trainset = trainset
        self.valset = valset

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.device = device
        self.epoch = epoch
        self.scale_data = scale_data

        self.use_ema = use_ema
        self.discount = discount
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps

        self.goal_window_size = goal_window_size
        self.window_size = window_size
        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size
        self.pred_last_action_only = pred_last_action_only
        self.diffusion_kde = diffusion_kde
        self.diffusion_kde_samples = diffusion_kde_samples
        self.goal_conditioned = goal_conditioned

        self.obs_context = deque(maxlen=self.obs_seq_len)
        self.action_counter = self.action_seq_size

        # used for eval loop
        self.curr_action_seq = None

        # EMA init
        if self.use_ema:
            self.ema_helper = ExponentialMovingAverage(self.model.parameters(), decay=self.decay)

        # KDE init (optional)
        self.kde = None
        if self.diffusion_kde:
            self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2)

    def reset(self):
        self.obs_context = deque(maxlen=self.obs_seq_len)
        self.action_counter = self.action_seq_size
        self.curr_action_seq = None

    # -----------------------
    # BaseAgent required APIs
    # -----------------------
    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:

        # scale data if necessarry, otherwise the scaler will return unchanged values
        self.model.train()

        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        loss = self.model.loss(action, state, goal)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_ema:
            self.ema_helper.update(self.model.parameters())

        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:
        self.model.eval()

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        loss = self.model.loss(action, state, goal)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        return float(loss.detach().cpu().item())

    # -----------------------
    # Training / eval loops
    # -----------------------
    def train_agent(self, global_step: int = 0):
        """
        Train loop for diffusion model
        """
        self.optimizer = hydra.utils.instantiate(self.optimization, params=self.model.parameters())

        train_loader = self.train_dataloader
        self.model.train()
        self.model.to(self.device)

        for i in range(self.epoch):
            epoch_loss = 0.0
            for batch in tqdm(train_loader, disable=True):
                obs = batch["obs"].to(self.device)
                action = batch["action"].to(self.device)
                goal = batch["goal"].to(self.device) if self.goal_conditioned else None

                loss_val = self.train_step(obs, action, goal)
                epoch_loss += loss_val
                global_step += 1

            epoch_loss /= len(train_loader)
            wandb.log({"train/loss": epoch_loss, "epoch": i}, step=global_step)

            # validation and checkpoint handled in BaseAgent hooks
            self.eval_agent(global_step=global_step, epoch=i)

        return global_step

    @torch.no_grad()
    def eval_agent(self, global_step: int = 0, epoch: int = 0):
        """
        Validation loop
        """
        val_loader = self.val_dataloader
        self.model.eval()

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        val_loss = 0.0
        for batch in tqdm(val_loader, disable=True):
            obs = batch["obs"].to(self.device)
            action = batch["action"].to(self.device)
            goal = batch["goal"].to(self.device) if self.goal_conditioned else None
            val_loss += self.evaluate(obs, action, goal)

        val_loss /= len(val_loader)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

    # -----------------------
    # Inference (guidance)
    # -----------------------
    def predict(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, extra_args=None) -> torch.Tensor:
        """
        Predict next action (one step), but the model actually predicts an action sequence.
        We cache the sequence and return it step-by-step.
        """
        obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        obs = self.scaler.scale_input(obs)
        self.obs_context.append(obs)
        input_state = torch.stack(tuple(self.obs_context), dim=1)  # [B, T, obs_dim]

        # Only resample a new action sequence when the cached sequence is exhausted
        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            if self.use_ema:
                self.ema_helper.store(self.model.parameters())
                self.ema_helper.copy_to(self.model.parameters())

            self.model.eval()

            extra_args = extra_args or {}
            model_pred = self.model(input_state, goal, **extra_args)

            # restore the previous model parameters
            if self.use_ema:
                self.ema_helper.restore(self.model.parameters())

            model_pred = self.scaler.inverse_scale_output(model_pred)
            self.curr_action_seq = model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()

    # -----------------------
    # I/O
    # -----------------------
    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        if self.use_ema:
            self.ema_helper = ExponentialMovingAverage(self.model.parameters(), decay=self.decay)
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))
