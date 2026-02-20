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
Implemented a training pipeline with artifact tracking:
- Equinox transformer policy with EMS/item self-attention + cross-attention.
- Forward rollout through `gfnx.utils.forward_rollout`.
- TB-style optimization path using canonical trajectory probability utilities:
   - `gfnx.utils.forward_rollout`
   - `gfnx.utils.forward_trajectory_log_probs`
- `optax` optimizer and train loop with periodic metric printing.
- Local experiment artifacts per run:
   - `config.json` (config snapshot),
   - `metrics.csv` (training curves),
   - `checkpoints/step_*.eqx` + `checkpoints/latest.eqx` (model + `logZ`).

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

`train.py` now loads numeric defaults from `config.base.yaml` in the repo root.
CLI flags override values from that config file.

To use a different base config file:

```bash
python train.py --config /path/to/config.yaml
```

Training artifacts are saved under `runs/<run-name>/` by default.

TensorBoard logs are also saved per run under:

```bash
runs/<run-name>/tensorboard/
```

Launch TensorBoard with `uv`:

```bash
uv run tensorboard --logdir runs
```

Or for a single run:

```bash
uv run tensorboard --logdir runs/<run-name>/tensorboard
```

Useful options:

```bash
python train.py --num-train-steps 1000 --checkpoint-every 200 --log-every 50 --output-dir runs --run-name exp-001
```

### GPU selection (`--device gpu`)
- `train.py` now sets `CUDA_VISIBLE_DEVICES` **before importing `jax`**.
- Use `--gpu-id` to choose the physical GPU index to expose to the process.
- Default GPU selection comes from `config.base.yaml` (`runtime.device` / `runtime.gpu_id`).
- Example (use physical GPU 1 only):

```bash
python train.py --device gpu --gpu-id 1 --num-train-steps 10 --num-envs 8 --max-num-items 10 --max-num-ems 30 --obs-num-ems 30
```

- Example with env default (equivalent to passing `--gpu-id 1`):

```bash
FLOW_DEFAULT_GPU_ID=1 python train.py --device gpu --num-train-steps 10 --num-envs 8 --max-num-items 10 --max-num-ems 30 --obs-num-ems 30
```

- Inside JAX, the selected visible GPU appears as `cuda:0` (expected after masking).

## Current Simplifications / Assumptions
1. **Backward policy is fixed deterministic LIFO**
   - one backward action only (undo-last).
2. **Observation encoder is pure flatten+MLP**
   - now replaced by a lightweight structure-aware transformer; still no advanced set-equivariant pretraining or specialized geometric encoder.
3. **Reward is utilization-only**
   - no extra penalties (e.g., stability, support, compactness).
4. **Training script is minimal**
   - no replay buffer and no external tracking backend (e.g., W&B/MLflow), but local checkpointing + CSV logging are implemented.
5. **Backward policy logits are fixed in training rollout info**
   - backward logits are currently deterministic for the LIFO backward action space.

## TODOs

### Medium-priority
All medium-priority TODOs are implemented:
1. ✅ **Policy architecture improved**
   - replaced plain MLP by structure-aware transformer with EMS/item interactions.
2. ✅ **Checkpointing and experiment logging**
   - model + `logZ` checkpoints, `metrics.csv`, and `config.json` snapshot.
3. ✅ **Evaluation script added**
   - `evaluate.py` collects utilization statistics and top-k samples (by utilization) with action histories.

### Optional extensions
4. **Alternative rewards / curriculum**
   - reward shaping schedules and harder item distributions.
5. **Ablations**
   - deterministic PB vs learned PB, dense vs sparse reward, observation variants.

## Quick Status
- Wrapper implemented and executable.
- Short smoke training run has been verified in this workspace.
- Medium-priority implementation work is completed.
- Remaining work is in optional experimentation extensions.

## Evaluation
- Added [evaluate.py](evaluate.py) to evaluate saved checkpoints.
- It loads run config + checkpoint, runs forward rollouts, computes utilization summary stats, and exports top-k samples.

Example:

```bash
python evaluate.py --run-dir runs/exp-001 --num-eval-batches 10 --num-eval-envs 128 --top-k 10
```

## Regression Tests
- Added [tests/test_reset_key_rotation.py](tests/test_reset_key_rotation.py) to protect against frozen-seed regressions.
- Covered checks:
   - different reset keys produce different initial observations.
- Added [tests/integration_reset_key_rotation.py](tests/integration_reset_key_rotation.py) for the compile-heavy train-step check:
   - `env_params.reset_key` rotates across JITted train steps.
- Added [tests/test_wrapper_invariants.py](tests/test_wrapper_invariants.py) for wrapper API invariants.
- Run with:

```bash
python -m unittest tests/test_reset_key_rotation.py -v
python -m unittest tests/test_wrapper_invariants.py -v
python -m unittest tests/integration_reset_key_rotation.py -v
```
