from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple


def _default_gpu_id_from_env() -> int:
    raw_value = os.environ.get("FLOW_DEFAULT_GPU_ID")
    if raw_value is None:
        return 0
    try:
        return int(raw_value)
    except ValueError:
        return 0


_DEFAULT_GPU_ID = _default_gpu_id_from_env()


def _bootstrap_runtime_config() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-id", type=int, default=_DEFAULT_GPU_ID)
    args, _ = parser.parse_known_args(sys.argv[1:])

    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    return args.device, args.gpu_id


_BOOTSTRAP_DEVICE, _BOOTSTRAP_GPU_ID = _bootstrap_runtime_config()

import chex
import equinox as eqx
import gfnx
import jax
import jax.numpy as jnp
import optax

from env_wrapper import BinPackEnvParams, BinPackGFN


class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        qk_size: int,
        ff_dim: int,
        *,
        key: chex.PRNGKey,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=model_dim,
            key_size=model_dim,
            value_size=model_dim,
            output_size=model_dim,
            qk_size=qk_size,
            vo_size=qk_size,
            key=k1,
        )
        self.norm1 = eqx.nn.LayerNorm(model_dim)
        self.ff_in = eqx.nn.Linear(model_dim, ff_dim, key=k2)
        self.ff_out = eqx.nn.Linear(ff_dim, model_dim, key=k3)
        self.norm2 = eqx.nn.LayerNorm(model_dim)

    def __call__(
        self,
        query: chex.Array,
        key_value: chex.Array,
        mask: chex.Array,
    ) -> chex.Array:
        attn_out = self.attention(query, key_value, key_value, mask=mask)
        h = jax.vmap(self.norm1)(query + attn_out)
        ff = jax.vmap(self.ff_out)(jax.nn.silu(jax.vmap(self.ff_in)(h)))
        return jax.vmap(self.norm2)(h + ff)


def _make_self_attention_mask(mask: chex.Array) -> chex.Array:
    base = jnp.logical_and(mask[:, None], mask[None, :])
    diagonal = jnp.eye(mask.shape[0], dtype=jnp.bool_)
    return jnp.logical_or(base, jnp.logical_and(jnp.logical_not(mask)[:, None], diagonal))


def _make_cross_attention_mask(mask: chex.Array) -> chex.Array:
    has_any = jnp.any(mask, axis=-1, keepdims=True)
    fallback = jnp.zeros_like(mask)
    fallback = fallback.at[:, 0].set(True)
    return jnp.where(has_any, mask, fallback)


def _infer_obs_structure(obs_dim: int, num_actions: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for num_ems in range(1, num_actions + 1):
        if num_actions % num_ems != 0:
            continue
        num_items = num_actions // num_ems
        if 7 * num_ems + 5 * num_items + num_actions == obs_dim:
            candidates.append((num_ems, num_items))
    if not candidates:
        raise ValueError(
            f"Could not infer (obs_num_ems, max_num_items) from obs_dim={obs_dim}, num_actions={num_actions}."
        )
    candidates.sort(reverse=True)
    return candidates[0]


class PolicyTransformer(eqx.Module):
    obs_num_ems: int = eqx.field(static=True)
    max_num_items: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    ems_projection: eqx.nn.Linear
    item_projection: eqx.nn.Linear
    self_ems_blocks: tuple[TransformerBlock, ...]
    self_item_blocks: tuple[TransformerBlock, ...]
    cross_ems_item_blocks: tuple[TransformerBlock, ...]
    cross_item_ems_blocks: tuple[TransformerBlock, ...]
    policy_ems_head: eqx.nn.Linear
    policy_item_head: eqx.nn.Linear
    flow_head: eqx.nn.Linear

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        *,
        obs_num_ems: int,
        max_num_items: int,
        key: chex.PRNGKey,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 2,
    ):
        expected_obs_dim = 7 * obs_num_ems + 5 * max_num_items + (obs_num_ems * max_num_items)
        if obs_dim != expected_obs_dim:
            raise ValueError(
                f"obs_dim mismatch: got {obs_dim}, expected {expected_obs_dim} for "
                f"obs_num_ems={obs_num_ems}, max_num_items={max_num_items}."
            )
        if num_actions != obs_num_ems * max_num_items:
            raise ValueError(
                f"num_actions mismatch: got {num_actions}, expected {obs_num_ems * max_num_items}."
            )

        self.obs_num_ems = obs_num_ems
        self.max_num_items = max_num_items
        self.num_actions = num_actions

        qk_size = max(1, hidden_dim // num_heads)
        ff_dim = ff_multiplier * hidden_dim

        num_keys = 5 + 4 * num_layers
        keys = jax.random.split(key, num_keys)
        self.ems_projection = eqx.nn.Linear(6, hidden_dim, key=keys[0])
        self.item_projection = eqx.nn.Linear(3, hidden_dim, key=keys[1])

        offset = 2
        self.self_ems_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )
        offset += num_layers
        self.self_item_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )
        offset += num_layers
        self.cross_ems_item_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )
        offset += num_layers
        self.cross_item_ems_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )

        self.policy_ems_head = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[-3])
        self.policy_item_head = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[-2])
        self.flow_head = eqx.nn.Linear(2 * hidden_dim, 1, key=keys[-1])

    def _parse_obs(
        self, obs: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        num_ems = self.obs_num_ems
        num_items = self.max_num_items

        ems_coords_end = 6 * num_ems
        ems_mask_end = ems_coords_end + num_ems
        item_feats_end = ems_mask_end + 3 * num_items
        item_mask_end = item_feats_end + num_items
        placed_end = item_mask_end + num_items
        action_mask_end = placed_end + (num_ems * num_items)

        ems_coords = obs[:ems_coords_end].reshape(num_ems, 6)
        ems_mask = obs[ems_coords_end:ems_mask_end] > 0.5
        item_feats = obs[ems_mask_end:item_feats_end].reshape(num_items, 3)
        items_mask = obs[item_feats_end:item_mask_end] > 0.5
        items_placed = obs[item_mask_end:placed_end] > 0.5
        action_mask = obs[placed_end:action_mask_end].reshape(num_ems, num_items) > 0.5

        return ems_coords, ems_mask, item_feats, items_mask, items_placed, action_mask

    def __call__(self, obs: chex.Array) -> tuple[chex.Array, chex.Array]:
        ems_coords, ems_mask, item_feats, items_mask, items_placed, action_mask = self._parse_obs(obs)
        valid_items = jnp.logical_and(items_mask, jnp.logical_not(items_placed))

        ems_embeddings = jax.vmap(self.ems_projection)(ems_coords)
        item_embeddings = jax.vmap(self.item_projection)(item_feats)
        ems_embeddings = jnp.where(ems_mask[:, None], ems_embeddings, 0.0)
        item_embeddings = jnp.where(valid_items[:, None], item_embeddings, 0.0)

        ems_self_mask = _make_self_attention_mask(ems_mask)
        item_self_mask = _make_self_attention_mask(valid_items)
        action_mask = jnp.logical_and(action_mask, ems_mask[:, None])
        action_mask = jnp.logical_and(action_mask, valid_items[None, :])
        ems_cross_items_mask = _make_cross_attention_mask(
            action_mask
        )
        items_cross_ems_mask = _make_cross_attention_mask(
            jnp.swapaxes(action_mask, 0, 1)
        )

        for self_ems, self_items, cross_ems_items, cross_items_ems in zip(
            self.self_ems_blocks,
            self.self_item_blocks,
            self.cross_ems_item_blocks,
            self.cross_item_ems_blocks,
        ):
            ems_embeddings = self_ems(ems_embeddings, ems_embeddings, ems_self_mask)
            item_embeddings = self_items(item_embeddings, item_embeddings, item_self_mask)
            new_ems_embeddings = cross_ems_items(
                ems_embeddings,
                item_embeddings,
                ems_cross_items_mask,
            )
            item_embeddings = cross_items_ems(
                item_embeddings,
                ems_embeddings,
                items_cross_ems_mask,
            )
            ems_embeddings = new_ems_embeddings

            ems_embeddings = jnp.where(ems_mask[:, None], ems_embeddings, 0.0)
            item_embeddings = jnp.where(valid_items[:, None], item_embeddings, 0.0)

        ems_policy = jax.vmap(self.policy_ems_head)(ems_embeddings)
        items_policy = jax.vmap(self.policy_item_head)(item_embeddings)
        logits_matrix = jnp.einsum("ek,ik->ei", ems_policy, items_policy)
        logits = logits_matrix.reshape(self.num_actions)

        ems_global = jnp.sum(jnp.where(ems_mask[:, None], ems_embeddings, 0.0), axis=0)
        items_global = jnp.sum(jnp.where(valid_items[:, None], item_embeddings, 0.0), axis=0)
        joint = jnp.concatenate([ems_global, items_global], axis=-1)
        log_flow = jnp.squeeze(self.flow_head(joint), axis=-1)
        return logits, log_flow


class PolicyMLP(PolicyTransformer):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int, *, key: chex.PRNGKey):
        obs_num_ems, max_num_items = _infer_obs_structure(obs_dim, num_actions)
        super().__init__(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            obs_num_ems=obs_num_ems,
            max_num_items=max_num_items,
            key=key,
        )


class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    env: BinPackGFN
    env_params: BinPackEnvParams
    model: Any
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    logZ: chex.Array
    num_envs: int


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return str(value)


def _create_run_dir(output_dir: str, run_name: str | None, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    resolved_run_name = run_name or f"run-{timestamp}-seed{seed}"
    run_dir = Path(output_dir) / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_checkpoint(path: Path, model: Any, logZ: chex.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = (model, jnp.asarray(logZ, dtype=jnp.float32))
    eqx.tree_serialise_leaves(path.as_posix(), checkpoint_payload)


def _resolve_device(device_kind: str) -> Any:
    if device_kind == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("No CPU device found by JAX.")
        return cpu_devices[0]

    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("No GPU device found by JAX.")
    return gpu_devices[0]


def trajectory_balance_loss(
    model: Any,
    logZ: chex.Array,
    traj_data: gfnx.utils.TrajectoryData,
    rollout_info: dict[str, chex.Array],
    env: BinPackGFN,
    env_params: BinPackEnvParams,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    del model
    log_pf_traj, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(env, traj_data, env_params)
    terminal_log_reward = rollout_info["log_gfn_reward"]

    residual = logZ + log_pf_traj - log_pb_traj - terminal_log_reward
    loss = jnp.mean(residual**2)
    metrics = {
        "loss": loss,
        "mean_terminal_log_reward": jnp.mean(terminal_log_reward),
        "mean_log_pf": jnp.mean(log_pf_traj),
        "mean_log_pb": jnp.mean(log_pb_traj),
        "mean_utilization": jnp.mean(terminal_log_reward / env_params.reward_params.beta),
    }
    return loss, metrics


def make_train_step():
    @eqx.filter_jit
    def train_step(train_state: TrainState) -> tuple[TrainState, dict[str, chex.Array]]:
        rng_key, rollout_key, new_env_key = jax.random.split(train_state.rng_key, 3)
        rollout_env_params = eqx.tree_at(
            lambda p: p.reset_key,
            train_state.env_params,
            new_env_key,
        )

        policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

        def fwd_policy_fn(
            fwd_rng_key: chex.PRNGKey,
            env_obs: chex.Array,
            current_policy_params: Any,
        ) -> tuple[chex.Array, dict[str, chex.Array]]:
            del fwd_rng_key
            model = eqx.combine(current_policy_params, policy_static)
            logits, _ = jax.vmap(model)(env_obs)
            backward_logits = jnp.zeros((logits.shape[0], train_state.env.backward_action_space.n))
            return logits, {
                "forward_logits": logits,
                "backward_logits": backward_logits,
            }

        traj_data, rollout_info = gfnx.utils.forward_rollout(
            rng_key=rollout_key,
            num_envs=train_state.num_envs,
            policy_fn=fwd_policy_fn,
            policy_params=policy_params,
            env=train_state.env,
            env_params=rollout_env_params,
        )

        def loss_fn(
            params: tuple[Any, chex.Array],
        ) -> tuple[chex.Array, dict[str, chex.Array]]:
            model_params, logZ = params
            model = eqx.combine(model_params, policy_static)
            return trajectory_balance_loss(
                model,
                logZ,
                traj_data,
                rollout_info,
                train_state.env,
                rollout_env_params,
            )

        params = (policy_params, train_state.logZ)
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
        del loss

        updates, opt_state = train_state.optimizer.update(grads, train_state.opt_state, params)
        new_model_params, logZ = eqx.apply_updates(params, updates)
        model = eqx.combine(new_model_params, policy_static)

        next_state = TrainState(
            rng_key=rng_key,
            env=train_state.env,
            env_params=rollout_env_params,
            model=model,
            optimizer=train_state.optimizer,
            opt_state=opt_state,
            logZ=logZ,
            num_envs=train_state.num_envs,
        )
        return next_state, metrics

    return train_step


def run_training(
    seed: int,
    num_train_steps: int,
    num_envs: int,
    learning_rate: float,
    hidden_dim: int,
    beta: float,
    max_num_items: int,
    max_num_ems: int,
    obs_num_ems: int,
    device: str,
    gpu_id: int,
    output_dir: str,
    run_name: str | None,
    checkpoint_every: int,
    log_every: int,
) -> None:
    run_dir = _create_run_dir(output_dir=output_dir, run_name=run_name, seed=seed)
    checkpoints_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"

    run_config = {
        "seed": seed,
        "num_train_steps": num_train_steps,
        "num_envs": num_envs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "beta": beta,
        "max_num_items": max_num_items,
        "max_num_ems": max_num_ems,
        "obs_num_ems": obs_num_ems,
        "device": device,
        "gpu_id": gpu_id,
        "output_dir": output_dir,
        "run_name": run_dir.name,
        "checkpoint_every": checkpoint_every,
        "log_every": log_every,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(config_path, run_config)

    metric_fieldnames = [
        "step",
        "loss",
        "mean_utilization",
        "mean_terminal_log_reward",
        "mean_log_pf",
        "mean_log_pb",
        "logZ",
    ]

    selected_device = _resolve_device(device)
    if device == "gpu":
        print(
            f"Using device: {selected_device} "
            f"(requested={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, gpu_id={gpu_id})"
        )
    else:
        print(f"Using device: {selected_device} (requested={device})")
    print(f"Artifacts directory: {run_dir}")

    with jax.default_device(selected_device):
        rng = jax.random.PRNGKey(seed)

        env = BinPackGFN(
            max_num_items=max_num_items,
            max_num_ems=max_num_ems,
            obs_num_ems=obs_num_ems,
            beta=beta,
            dense_reward=False,
        )
        rng, env_init_key, net_init_key = jax.random.split(rng, 3)
        env_params = env.init(env_init_key)

        obs_dim = env.observation_space["shape"][0]
        n_actions = env.action_space.n
        model = PolicyTransformer(
            obs_dim=obs_dim,
            num_actions=n_actions,
            hidden_dim=hidden_dim,
            obs_num_ems=obs_num_ems,
            max_num_items=max_num_items,
            key=net_init_key,
        )

        optimizer = optax.adam(learning_rate)
        params = (eqx.filter(model, eqx.is_array), jnp.asarray(0.0, dtype=jnp.float32))
        opt_state = optimizer.init(params)

        train_state = TrainState(
            rng_key=rng,
            env=env,
            env_params=env_params,
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            logZ=jnp.asarray(0.0, dtype=jnp.float32),
            num_envs=num_envs,
        )

        train_step = make_train_step()

        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=metric_fieldnames)
            writer.writeheader()

            for step in range(num_train_steps):
                train_state, metrics = train_step(train_state)

                row = {
                    "step": step,
                    "loss": float(metrics["loss"]),
                    "mean_utilization": float(metrics["mean_utilization"]),
                    "mean_terminal_log_reward": float(metrics["mean_terminal_log_reward"]),
                    "mean_log_pf": float(metrics["mean_log_pf"]),
                    "mean_log_pb": float(metrics["mean_log_pb"]),
                    "logZ": float(train_state.logZ),
                }
                writer.writerow({key: _json_safe_value(value) for key, value in row.items()})

                if step % log_every == 0 or step == num_train_steps - 1:
                    print(
                        f"step={step:05d} "
                        f"loss={row['loss']:.6f} "
                        f"mean_utilization={row['mean_utilization']:.6f} "
                        f"mean_terminal_log_reward={row['mean_terminal_log_reward']:.6f} "
                        f"logZ={row['logZ']:.6f}"
                    )

                should_checkpoint = ((step + 1) % checkpoint_every == 0) or (step == num_train_steps - 1)
                if should_checkpoint:
                    ckpt_name = f"step_{step + 1:06d}.eqx"
                    ckpt_path = checkpoints_dir / ckpt_name
                    _save_checkpoint(ckpt_path, train_state.model, train_state.logZ)
                    _save_checkpoint(checkpoints_dir / "latest.eqx", train_state.model, train_state.logZ)
                    _write_json(
                        checkpoints_dir / "latest.json",
                        {
                            "step": step + 1,
                            "logZ": float(train_state.logZ),
                            "checkpoint": ckpt_name,
                        },
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TB GFlowNet on Jumanji BinPack.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--max-num-items", type=int, default=20)
    parser.add_argument("--max-num-ems", type=int, default=40)
    parser.add_argument("--obs-num-ems", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory where run artifacts (config/metrics/checkpoints) are saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to timestamped name when omitted.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save checkpoints every N train steps and at final step.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training metrics every N train steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Execution device. Use 'gpu' to enable CUDA via JAX.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=_BOOTSTRAP_GPU_ID,
        help=(
            "Physical GPU index to expose via CUDA_VISIBLE_DEVICES. "
            "Applied before importing jax. Default: FLOW_DEFAULT_GPU_ID or 0."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    run_training(
        seed=args.seed,
        num_train_steps=args.num_train_steps,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        max_num_items=args.max_num_items,
        max_num_ems=args.max_num_ems,
        obs_num_ems=args.obs_num_ems,
        device=args.device,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        run_name=args.run_name,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
    )
