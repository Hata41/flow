from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

import equinox as eqx
import gfnx
import jax
import jax.numpy as jnp
import numpy as np

from env_wrapper import BinPackGFN
from training_model import PolicyTransformer


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _serialize_action_history(action_history: np.ndarray, step_count: int) -> list[list[int]]:
    valid_steps = max(0, int(step_count))
    prefix = action_history[:valid_steps]
    return [[int(step[0]), int(step[1])] for step in prefix]


def _build_eval_summary(
    utilizations: list[float],
    trajectories: list[dict[str, Any]],
    top_k: int,
    run_dir: Path,
    checkpoint_path: Path,
    logZ: float,
    eval_args: argparse.Namespace,
) -> dict[str, Any]:
    util_array = np.asarray(utilizations, dtype=np.float32)
    if util_array.size == 0:
        raise RuntimeError("No evaluation samples were collected.")

    ranked = sorted(trajectories, key=lambda item: item["utilization"], reverse=True)
    top_samples = ranked[:top_k]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": run_dir.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "logZ": float(logZ),
        "num_samples": int(util_array.size),
        "summary": {
            "mean_utilization": float(np.mean(util_array)),
            "std_utilization": float(np.std(util_array)),
            "min_utilization": float(np.min(util_array)),
            "max_utilization": float(np.max(util_array)),
            "p50_utilization": float(np.percentile(util_array, 50)),
            "p90_utilization": float(np.percentile(util_array, 90)),
            "p95_utilization": float(np.percentile(util_array, 95)),
            "mean_trajectory_length": float(np.mean([row["step_count"] for row in trajectories])),
        },
        "top_k": int(top_k),
        "top_samples": top_samples,
        "eval_config": {
            "seed": int(eval_args.seed),
            "device": eval_args.device,
            "gpu_id": int(eval_args.gpu_id),
            "num_eval_batches": int(eval_args.num_eval_batches),
            "num_eval_envs": int(eval_args.num_eval_envs),
        },
    }


def _resolve_model_config(run_config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = run_config.get("model")
    if isinstance(model_cfg, dict):
        resolved = dict(model_cfg)
    else:
        resolved = {}

    if "hidden_dim" in run_config:
        resolved["hidden_dim"] = int(run_config["hidden_dim"])

    resolved.setdefault("num_layers", 2)
    resolved.setdefault("num_heads", 4)
    resolved.setdefault("ff_multiplier", 2)
    resolved.setdefault("qk_size_min", 1)
    resolved.setdefault("obs_ems_feature_factor", 7)
    resolved.setdefault("obs_item_feature_factor", 5)
    resolved.setdefault("ems_coord_dim", 6)
    resolved.setdefault("item_feature_dim", 3)
    resolved.setdefault("mask_threshold", 0.5)
    resolved.setdefault("key_count_base", 5)
    resolved.setdefault("key_count_per_layer", 4)
    resolved.setdefault("key_offset_initial", 2)
    resolved.setdefault("flow_input_multiplier", 2)
    resolved.setdefault("flow_output_dim", 1)
    resolved.setdefault("policy_ems_head_key_index", -3)
    resolved.setdefault("policy_item_head_key_index", -2)
    resolved.setdefault("flow_head_key_index", -1)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BinPack GFlowNet checkpoint.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory produced by train.py.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoints/latest.eqx.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-batches", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <run-dir>/eval/eval-<timestamp>.json.",
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


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.num_eval_batches <= 0:
        raise ValueError("--num-eval-batches must be > 0")
    if args.num_eval_envs <= 0:
        raise ValueError("--num-eval-envs must be > 0")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    run_config = _load_json(config_path)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "checkpoints" / "latest.eqx")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    selected_device = _resolve_device(args.device)
    if args.device == "gpu":
        print(
            f"Using device: {selected_device} "
            f"(requested={args.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, gpu_id={args.gpu_id})"
        )
    else:
        print(f"Using device: {selected_device} (requested={args.device})")

    with jax.default_device(selected_device):
        max_num_items = int(run_config["max_num_items"])
        max_num_ems = int(run_config["max_num_ems"])
        obs_num_ems = int(run_config["obs_num_ems"])
        model_cfg = _resolve_model_config(run_config)
        hidden_dim = int(model_cfg["hidden_dim"])
        beta = float(run_config["beta"])

        env = BinPackGFN(
            max_num_items=max_num_items,
            max_num_ems=max_num_ems,
            obs_num_ems=obs_num_ems,
            beta=beta,
            dense_reward=False,
        )

        rng = jax.random.PRNGKey(args.seed)
        rng, env_init_key, model_init_key = jax.random.split(rng, 3)
        env_params = env.init(env_init_key)

        obs_dim = env.observation_space["shape"][0]
        n_actions = env.action_space.n

        model_template = PolicyTransformer(
            obs_dim=obs_dim,
            num_actions=n_actions,
            hidden_dim=hidden_dim,
            obs_num_ems=obs_num_ems,
            max_num_items=max_num_items,
            key=model_init_key,
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            ff_multiplier=int(model_cfg["ff_multiplier"]),
            qk_size_min=int(model_cfg["qk_size_min"]),
            obs_ems_feature_factor=int(model_cfg["obs_ems_feature_factor"]),
            obs_item_feature_factor=int(model_cfg["obs_item_feature_factor"]),
            ems_coord_dim=int(model_cfg["ems_coord_dim"]),
            item_feature_dim=int(model_cfg["item_feature_dim"]),
            mask_threshold=float(model_cfg["mask_threshold"]),
            key_count_base=int(model_cfg["key_count_base"]),
            key_count_per_layer=int(model_cfg["key_count_per_layer"]),
            key_offset_initial=int(model_cfg["key_offset_initial"]),
            flow_input_multiplier=int(model_cfg["flow_input_multiplier"]),
            flow_output_dim=int(model_cfg["flow_output_dim"]),
            policy_ems_head_key_index=int(model_cfg["policy_ems_head_key_index"]),
            policy_item_head_key_index=int(model_cfg["policy_item_head_key_index"]),
            flow_head_key_index=int(model_cfg["flow_head_key_index"]),
        )
        model, logZ = eqx.tree_deserialise_leaves(
            checkpoint_path.as_posix(),
            (model_template, jnp.asarray(0.0, dtype=jnp.float32)),
        )

        policy_params, policy_static = eqx.partition(model, eqx.is_array)

        def policy_fn(
            fwd_rng_key: jax.Array,
            env_obs: jax.Array,
            current_policy_params: Any,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            del fwd_rng_key
            current_model = eqx.combine(current_policy_params, policy_static)
            logits, _ = jax.vmap(current_model)(env_obs)
            backward_logits = jnp.zeros((logits.shape[0], env.backward_action_space.n))
            return logits, {
                "forward_logits": logits,
                "backward_logits": backward_logits,
            }

        utilizations: list[float] = []
        trajectories: list[dict[str, Any]] = []

        for batch_idx in range(args.num_eval_batches):
            rng, rollout_key, new_env_key = jax.random.split(rng, 3)
            rollout_env_params = eqx.tree_at(lambda p: p.reset_key, env_params, new_env_key)

            _, rollout_info = gfnx.utils.forward_rollout(
                rng_key=rollout_key,
                num_envs=args.num_eval_envs,
                policy_fn=policy_fn,
                policy_params=policy_params,
                env=env,
                env_params=rollout_env_params,
            )

            batch_util = jax.device_get(rollout_info["log_gfn_reward"] / rollout_env_params.reward_params.beta)
            final_state = rollout_info["final_env_state"]
            batch_action_history = jax.device_get(final_state.action_history)
            batch_step_count = jax.device_get(final_state.step_count)

            for env_idx in range(args.num_eval_envs):
                util = float(batch_util[env_idx])
                step_count = int(batch_step_count[env_idx])
                action_history = _serialize_action_history(batch_action_history[env_idx], step_count)
                sample = {
                    "batch_index": int(batch_idx),
                    "env_index": int(env_idx),
                    "utilization": util,
                    "step_count": step_count,
                    "action_history": action_history,
                }
                trajectories.append(sample)
                utilizations.append(util)

    summary = _build_eval_summary(
        utilizations=utilizations,
        trajectories=trajectories,
        top_k=args.top_k,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        logZ=float(logZ),
        eval_args=args,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "eval" / f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    stats = summary["summary"]
    print(f"Evaluation samples: {summary['num_samples']}")
    print(
        "Utilization stats: "
        f"mean={stats['mean_utilization']:.6f} "
        f"std={stats['std_utilization']:.6f} "
        f"p90={stats['p90_utilization']:.6f} "
        f"max={stats['max_utilization']:.6f}"
    )
    print(f"Saved evaluation report: {output_path}")


if __name__ == "__main__":
    main()
