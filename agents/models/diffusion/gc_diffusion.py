# agents/models/diffusion/gc_diffusion.py
from matplotlib.backend_bases import MouseEvent
from typing import Optional, Callable, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from omegaconf import DictConfig
import hydra

from .utils import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
)

# code adapted from https://github.com/twitter/diffusion-rl/blob/master/agents/diffusion.py
class Diffusion(nn.Module):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            model: DictConfig,
            beta_schedule: str,
            n_timesteps: int,
            loss_type: str,
            clip_denoised: bool,
            predict_epsilon=True,
            device: str = 'cuda',
            diffusion_x: bool = False,
            diffusion_x_M: int = 32,
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = None
        # chose your beta style
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'vp':
            # beta max: 10 beta min: 0.1
            self.betas = vp_beta_schedule(n_timesteps).to(self.device)

        self.model = hydra.utils.instantiate(model)
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # define alpha stuff
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        # required for forward diffusion q( x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1).to(self.device)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(self.device)

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20)).to(self.device)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(self.device)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod).to(self.device)

        # either l1 or l2
        self.loss_fn = Losses[loss_type]()

        self.diffusion_x = diffusion_x
        self.diffusion_x_M = diffusion_x_M

    def get_params(self):
        return self.model.get_params()

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor, grad: bool = True):
        if grad:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s, g))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model_frozen(x, t, s, g))

        if self.clip_denoised:
            x_recon.clamp_(self.min_action, self.max_action)
        else:
            raise RuntimeError("clip_denoised must be True in this repo setup")

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor, grad: bool = True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, g=g, grad=grad)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def guided_p_sample_external(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        guidance_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        guidance_scale: float,
        grad_clip_norm: Optional[float] = None,
        grad_normalize: bool = False,
        guidance_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        One reverse diffusion step with external gradient guidance:
        mean <- mean - guidance_scale * var * grad_J

        guidance_fn must return grad_J w.r.t. action (same shape as x).
        """
        guidance_kwargs = guidance_kwargs or {}
        b, *_ = x.shape

        model_mean, posterior_variance, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, g=g, grad=True)

        # grad_J in action space
        grad_J = guidance_fn(s, model_mean, t, **guidance_kwargs)
        if not torch.is_tensor(grad_J):
            grad_J = torch.as_tensor(grad_J, device=self.device, dtype=model_mean.dtype)
        grad_J = grad_J.to(device=model_mean.device, dtype=model_mean.dtype)

        if grad_normalize:
            denom = grad_J.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
            grad_J = grad_J / denom.view(b, *((1,) * (len(grad_J.shape) - 1)))

        if grad_clip_norm is not None:
            gnorm = grad_J.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
            scale = (grad_clip_norm / gnorm).clamp_max(1.0)
            grad_J = grad_J * scale.view(b, *((1,) * (len(grad_J.shape) - 1)))

        # mean shift
        guided_mean = model_mean - guidance_scale * posterior_variance * grad_J

        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        x_prev = guided_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_prev

    def p_sample_loop(
        self,
        state,
        goal,
        shape,
        verbose: bool = False,
        return_diffusion: bool = False,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        guidance_scale: float = 0.0,
        guidance_start: float = 0.2,
        guidance_every: int = 1,
        grad_clip_norm: Optional[float] = None,
        grad_normalize: bool = False,
        guidance_kwargs: Optional[Dict[str, Any]] = None,
    ):
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        if return_diffusion:
            diffusion = [x]

        # when to start applying guidance (same semantics as your guided_sample: only in last fraction)
        i_start = int(self.n_timesteps * guidance_start)

        for step_i, i in enumerate(reversed(range(0, self.n_timesteps))):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            use_guidance = (
                (guidance_fn is not None)
                and (guidance_scale is not None)
                and (guidance_scale > 0.0)
                and (i <= i_start)
                and (guidance_every > 0)
                and ((step_i % guidance_every) == 0)
            )

            if use_guidance:
                x = self.guided_p_sample_external(
                    x=x,
                    t=timesteps,
                    s=state,
                    g=goal,
                    guidance_fn=guidance_fn,
                    guidance_scale=float(guidance_scale),
                    grad_clip_norm=grad_clip_norm,
                    grad_normalize=grad_normalize,
                    guidance_kwargs=guidance_kwargs,
                )
            else:
                x = self.p_sample(x, timesteps, state, goal)

            if return_diffusion:
                diffusion.append(x)

        # diffusion_x extra iterations at t=0 
        if self.diffusion_x:
            timesteps = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
            for m in range(0, self.diffusion_x_M):
                if (guidance_fn is not None) and (guidance_scale > 0.0):
                    x = self.guided_p_sample_external(
                        x=x,
                        t=timesteps,
                        s=state,
                        g=goal,
                        guidance_fn=guidance_fn,
                        guidance_scale=float(guidance_scale),
                        grad_clip_norm=grad_clip_norm,
                        grad_normalize=grad_normalize,
                        guidance_kwargs=guidance_kwargs,
                    )
                else:
                    x = self.p_sample(x, timesteps, state, goal)
                if return_diffusion:
                    diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    
    def sample(self, state, goal, *args, **kwargs):
        """
        Main Method to generate actions conditioned on the batch of state inputs

        :param state: the current state observation to conditon the diffusion model

        :return: x_{0} the predicted actions from the diffusion model
        """
        batch_size = state.shape[0]
        if len(state.shape) == 3:
            shape = (batch_size, state.shape[1], self.action_dim)
        else:
            shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, goal, shape, *args, **kwargs)
        return action.clamp_(self.min_action, self.max_action)

    def guided_p_sample(self, x, t, s, g, fun):
        '''
        Sample x_{t-1} from the model at the given timestep with additional conditioning
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param s: the current state observation
        :param fun: the conditoning model, subclass from torch.nn.Model

        :return: x_{t-1}
        '''
        b, *_ = x.shape
        with torch.no_grad():
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, g=g)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Involve Function Guidance
        a = model_mean.clone().requires_grad_(True)
        q_value = fun(s, a)
        # q_value = q_value / q_value.abs().mean().detach()  # normalize q
        grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
        return (model_mean + model_log_variance * grads) + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def guided_sample(self, state: torch.Tensor, cond_fun, start: int = 0.2, verbose=False, return_diffusion=False):
        """
        Generated diffusion samples conditioned on cond_fun and starts deffusion at 20% of initial timesteps
        for faster generating process

        :param state: the current state batch
        :param cond_fun: the condition function to guide samples
        :param start:  defines the starting timestep 0 refers to starting sampling from t and 0.2 means
                       starting sampling from at 20% of the inital time step for faster sampling
        :return_diffusion: bool to decide, if only x_0 or all samples should be returned from the looop

        :return: either the predicted x_0 sample or a list with [x_{t-1}, .., x_{0}]
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        x = torch.randn(shape, device=self.device)
        i_start = self.n_timesteps * start

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i <= i_start:
                x = self.guided_p_sample(x, timesteps, state, cond_fun)
            else:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)

            if return_diffusion: diffusion.append(x)

        x = x.clamp_(self.min_action, self.max_action)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1).clamp_(self.min_action, self.max_action)
        else:
            return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, t: torch.Tensor, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state, goal)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def loss(self, x: torch.Tensor, state: torch.Tensor, goal: Optional[torch.Tensor] = None, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, goal, t, weights)

    def forward(self, state, goal, *args, **kwargs):
        return self.sample(state, goal, *args, **kwargs)

    def sample_t_middle(self, state: torch.Tensor, goal):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        x = torch.randn(shape, device=self.device)

        t = np.random.randint(0, int(self.n_timesteps * 0.2))
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, goal, grad=(i == t))
        action = x
        return action.clamp_(self.min_action, self.max_action)

    def sample_t_last(self, state: torch.Tensor, goal):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        x = torch.randn(shape, device=self.device)
        cur_T = np.random.randint(int(self.n_timesteps * 0.8), self.n_timesteps)
        for i in reversed(range(0, cur_T)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i != 0:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state, goal)
            else:
                x = self.p_sample(x, timesteps, state, goal)

        action = x
        return action.clamp_(self.min_action, self.max_action)

    def sample_last_few(self, state: torch.Tensor, goal):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        x = torch.randn(shape, device=self.device)
        nest_limit = 5
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i >= nest_limit:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state, goal)
            else:
                x = self.p_sample(x, timesteps, state, goal)

        action = x
        return action.clamp_(self.min_action, self.max_action)


