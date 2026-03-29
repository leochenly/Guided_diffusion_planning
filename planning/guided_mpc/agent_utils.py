from __future__ import annotations

import os
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf


def register_omegaconf_resolvers() -> None:
    def add_resolver(*args: Any):
        vals = []
        for a in args:
            try:
                vals.append(float(a))
            except Exception:
                vals.append(a)
        if all(isinstance(v, (int, float)) for v in vals):
            s = sum(vals)
            if abs(s - int(s)) < 1e-9:
                return int(s)
            return s
        return vals

    try:
        OmegaConf.register_new_resolver("add", add_resolver)
    except Exception:
        try:
            OmegaConf.register_resolver("add", add_resolver)
        except Exception:
            pass


def _force_int_fields(cfg) -> None:
    int_keys = [
        "window_size",
        "obs_dim",
        "action_dim",
        "n_layer",
        "n_head",
        "n_embd",
        "train_batch_size",
        "val_batch_size",
        "num_workers",
        "epoch",
        "eval_every_n_epochs",
        "max_len_data",
    ]
    for k in int_keys:
        try:
            v = OmegaConf.select(cfg, k)
            if v is None:
                continue
            if isinstance(v, float) and abs(v - int(v)) < 1e-9:
                OmegaConf.update(cfg, k, int(v), merge=False)
        except Exception:
            pass


def load_agent(exp_dir: str, checkpoint_name: str, device: str = "cuda"):
    """Load the original Hydra agent in a reusable way."""
    config_path = os.path.join(exp_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    register_omegaconf_resolvers()
    cfg = OmegaConf.load(config_path)
    _force_int_fields(cfg)

    if "device" in cfg:
        cfg.device = device
    if "agents" in cfg and "device" in cfg.agents:
        cfg.agents.device = device
    if "agents" in cfg and "model" in cfg.agents and "device" in cfg.agents.model:
        cfg.agents.model.device = device

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(weights_path=exp_dir, sv_name=checkpoint_name)

    if device == "cpu":
        agent.model.to(torch.device("cpu"))
        if hasattr(agent, "ema_helper") and agent.ema_helper is not None:
            agent.ema_helper.device = torch.device("cpu")

    agent.model.eval()
    return agent
