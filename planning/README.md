# Guided MPC Diffusion 

A lightweight, modular refactor of a **guided diffusion planner** for the D3IL Avoiding task, with **MJX/JAX-based differentiable rollout guidance**.

This repository reorganizes the original research code into reusable modules so that other users can:

- load a pretrained diffusion policy,
- run a differentiable MJX rollout,
- define rollout-based costs,
- inject cost gradients into the denoising process,
- evaluate the planner in the Avoiding environment,
- visualize trajectories and denoising statistics.

The current version contains only the core guidance terms:

- **center-x cost**
- **optional barrier cost**

---

## 1. Project overview

The planner combines two ideas:

1. **Diffusion policy as a trajectory prior**  
   A pretrained diffusion model proposes a future sequence of end-effector motion increments.

2. **Differentiable rollout as guidance**  
   The sampled trajectory is rolled out through an MJX/JAX-based differentiable surrogate of the control-and-motion pipeline. A rollout-based cost is evaluated, differentiated with respect to the trajectory variable, and used to guide the reverse diffusion process.


---

## 2. Repository structure

The planning script layout is:

```text
planning/
├── guided_mpc/
│   ├── __init__.py
│   ├── constants.py
│   ├── costs.py
│   ├── simulator.py
│   ├── planner.py
│   ├── plots.py
│   └── agent_utils.py
├── examples/
│   └── run_guided_avoiding.py
└── README.md
```

### `guided_mpc/constants.py`
Stores default constants and small shared settings used by the other modules.

Typical content:

- default horizon length
- rollout step settings
- default cost weights
- numerical stability parameters

### `guided_mpc/costs.py`
Defines the rollout-based objective functions.

This file should contain the cost terms that are differentiated with respect to the trajectory variable.

Typical content:

- center-x cost
- smoothness cost
- barrier cost
- guidance configuration dataclass / config helpers
- cost aggregation utilities

### `guided_mpc/simulator.py`
Contains the differentiable MJX rollout wrapper.

This module is responsible for:

- copying the current environment state into MJX-compatible state
- integrating future increments into target setpoints
- executing the controller + MJX rollout loop
- returning the rollout TCP trajectory used by the cost function

### `guided_mpc/planner.py`
Contains the main guided diffusion planner.

This module is responsible for:

- receiving a diffusion model sample
- calling differentiable rollout
- computing trajectory cost
- differentiating cost with respect to the planned sequence
- injecting the gradient into the denoising process

### `guided_mpc/plots.py`
Provides plotting utilities for trajectory and denoising analysis.

Typical plots include:

- rollout trajectories
- center-x cost statistics
- gradient norm curves
- denoising step cost curves
- episode-level summaries

### `guided_mpc/agent_utils.py`
Contains utilities for loading or wrapping the pretrained diffusion agent.

Typical content:

- checkpoint loading
- Hydra/config loading helpers
- agent construction wrappers
- device placement helpers

### `examples/run_guided_avoiding.py`
A runnable example that wires everything together in a task script.

This file should be the place for:

- environment creation
- checkpoint path definition
- experiment hyperparameters
- success-rate evaluation
- trajectory plotting

The key principle is:

> reusable code goes in `guided_mpc/`, experiment-specific code goes in `examples/`.

---

## 3. Cost terms 

### 5.1 Center-x cost

Encourages the rollout trajectory to stay close to the center corridor:

### 5.2 Barrier cost

Applies a differentiable soft penalty around forbidden regions or geometric barriers.

Typical usage:

- rectangular keep-out region
- line-segment barrier
- softplus-style distance penalty

---

## 4. Running ablations

A recommended experiment is to compare different guidance strengths:

- `alpha = 0`
- `alpha = 15`
- `alpha = 50`

What to compare:

- trajectory plotting
- success rate
- mean executed cost
- gradient norm vs diffusion step
- completion time

---
