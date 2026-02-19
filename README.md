# 3D Bin Packing with Jumanji + gfnx

## Objective
Train a GFlowNet agent (Trajectory Balance style) to generate high-utilization 3D bin-packing configurations by:
- using `jumanji` for environment dynamics (physics/packing constraints), and
- using `gfnx` for trajectory rollouts and GFlowNet training scaffolding.

The core challenge is bridging API and semantics differences between the two libraries.

## What Has Been Implemented

### Project Structure
- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)
- [env_wrapper.py](env_wrapper.py)
- [train.py](train.py)

### Environment Wrapper (`env_wrapper.py`)
Implemented `BinPackGFN`, a `gfnx.base.BaseVecEnvironment` wrapper around Jumanji BinPack, including:

1. **State wrapper**
   - `GFNBinPackState` extends `BaseEnvState` and stores:
     - native Jumanji state,
     - `is_terminal`, `is_initial`, `is_pad`,
     - `step_count`,
     - fixed-size JIT-safe `action_history: int32[max_items, 2]` with `-1` padding,
     - reset key (`init_key`) for deterministic re-simulation,
     - cached `volume_utilization`.

2. **Observation flattening (MLP-ready)**
   - `get_obs` returns a batched 2D array `[batch, obs_dim]`.
   - Each per-state observation is flattened into a single `float32` vector:
     - EMS coordinates (visible EMS only),
     - EMS mask,
     - item dimensions,
     - item mask,
     - item placed mask.

3. **Action flattening / unflattening**
   - Forward action space is flat discrete size `obs_num_ems * max_num_items`.
   - Conversion:
     - `ems_id = action // max_num_items`
     - `item_id = action % max_num_items`

4. **Mask semantics alignment**
   - Jumanji: `action_mask=True` means **valid**.
   - gfnx expects invalid-mask (`True` means **invalid**).
   - Implemented `get_invalid_mask` as `~valid_mask`, flattened row-major.

5. **Forward transition**
   - `_single_transition` unflattens action, calls Jumanji `step`, updates wrapper state, writes action to `action_history[step_count]`, increments `step_count`, updates utilization and terminal flags.

6. **Backward transition (deterministic LIFO)**
   - `_single_backward_transition` is implemented via re-simulation:
     - removes last action logically (`target_steps = step_count - 1`),
     - resets env from `init_key`,
     - replays prefix actions with `jax.lax.scan` up to `target_steps`,
     - rebuilds exact EMS geometry and state.

7. **Deterministic backward masking**
   - Backward action space is a single dummy action (`n=1`).
   - `get_invalid_backward_mask` marks it valid iff `step_count > 0`.
   - This enforces deterministic LIFO parent mapping (`P_B(parent|child)=1`).

8. **Reward transformation**
   - Uses transformed reward objective:
     - `R(x) = exp(beta * utilization)`.
   - Implemented in log-space for TB compatibility:
     - `log R(x) = beta * utilization`.

### Training Script (`train.py`)
Implemented a minimal training pipeline:
- Equinox MLP policy over flattened observations.
- Forward rollout through `gfnx.utils.forward_rollout`.
- TB-style optimization path with:
  - dynamic resolver for gfnx TB loss (if available),
  - fallback manual TB-like loss (`logZ + sum(log P_F) - log R`)² over sampled trajectories.
- `optax` optimizer and train loop with periodic metric printing.

## Runtime / Environment Notes

### Current Python Environment
Use Python 3.11 or 3.12 (Python 3.13 is not supported for this dependency set).

### Important dependency caveat
Python 3.13 is currently unsatisfiable here due upstream dependency constraints between:
- `gfnx` transitive constraints (notably `numpy==1.26.4`), and
- `dm-tree` on Python 3.13 (`numpy>=2.1.0`).

Use Python 3.11/3.12 for reproducible installs.

### Installation sequence that worked in this workspace
The following sequence is reproducible with Python 3.12:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match git+https://github.com/instadeepai/jumanji.git@main
uv pip install --index-strategy unsafe-best-match --no-deps git+https://github.com/d-tiapkin/gfnx.git@main
uv pip install --index-strategy unsafe-best-match equinox optax jaxtyping flashbax omegaconf networkx
```

Then run:

```bash
python train.py --num-train-steps 10 --num-envs 8 --max-num-items 10 --max-num-ems 30 --obs-num-ems 30
```

## Current Simplifications / Assumptions
1. **Backward policy is fixed deterministic LIFO**
   - one backward action only (undo-last).
2. **Observation encoder is pure flatten+MLP**
   - no permutation-equivariant/set encoder yet.
3. **Reward is utilization-only**
   - no extra penalties (e.g., stability, support, compactness).
4. **Training script is minimal**
   - no replay buffer, no checkpointing, no rich logging backend.
5. **TB loss integration fallback**
   - uses manual TB-like objective when direct `TrajectoryBalance` API is unavailable/unstable due version drift.

## TODOs

### High-priority
1. **Stabilize dependency lock for reproducibility**
   - either:
     - standardize on Python 3.12 and fully solvable lockfile, or
     - vendor/fork-compatible `gfnx` constraints for Python 3.13.
2. **Pin and verify exact gfnx TB API**
   - remove dynamic import fallback and call canonical TB loss directly.
3. **Add unit tests for wrapper invariants**
   - flatten/unflatten bijection,
   - forward mask inversion,
   - backward deterministic validity mask,
   - backward re-simulation consistency.

### Medium-priority
4. **Improve policy architecture**
   - replace plain MLP with structure-aware encoder (EMS-item interaction model / attention).
5. **Add checkpointing and experiment logging**
   - save model/logZ, training curves, config snapshot.
6. **Add evaluation script**
   - collect utilization statistics and top-k packing samples.

### Optional extensions
7. **Alternative rewards / curriculum**
   - reward shaping schedules and harder item distributions.
8. **Ablations**
   - deterministic PB vs learned PB, dense vs sparse reward, observation variants.

## Quick Status
- Wrapper implemented and executable.
- Short smoke training run has been verified in this workspace.
- Main remaining blocker is dependency reproducibility across environments, not core algorithmic wiring.

## Regression Tests
- Added [tests/test_reset_key_rotation.py](tests/test_reset_key_rotation.py) to protect against frozen-seed regressions.
- Covered checks:
   - `env_params.reset_key` rotates across JITted train steps.
   - different reset keys produce different initial observations.
- Run with:

```bash
python -m unittest tests/test_reset_key_rotation.py -v
```
