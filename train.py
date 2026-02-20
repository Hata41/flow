from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_config import (
    TrainingConfig,
    bootstrap_runtime_from_argv,
    resolve_config,
)


_BOOTSTRAP_DEVICE, _BOOTSTRAP_GPU_ID, _BOOTSTRAP_CONFIG_PATH = bootstrap_runtime_from_argv(sys.argv[1:])

import chex
import equinox as eqx
import gfnx
import jax
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter

from env_wrapper import BinPackGFN
from training_core import action_history_hamming_distance, make_fwd_policy_fn, make_train_step, TrainState
from training_model import build_policy_transformer_from_config


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


def _write_json(path: Path, payload: dict[str, Any], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
        handle.write("\n")


def _save_checkpoint(path: Path, model: Any, logZ: chex.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = (model, jnp.asarray(logZ, dtype=jnp.float32))
    eqx.tree_serialise_leaves(path.as_posix(), checkpoint_payload)


def _supports_color_output() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _colorize(text: str, color_code: int) -> str:
    if not _supports_color_output():
        return text
    return f"\033[{color_code}m{text}\033[0m"


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


def _format_train_log(message: str, train_log_color_code: int) -> str:
    return _colorize(f"[train] {message}", train_log_color_code)


def _format_eval_log(message: str, eval_log_color_code: int) -> str:
    return _colorize(f"[eval]  {message}", eval_log_color_code)


def _table_header(columns: list[tuple[str, int]]) -> str:
    return " ".join(f"{name:>{width}}" for name, width in columns)


def _table_separator(columns: list[tuple[str, int]]) -> str:
    return " ".join("-" * width for _, width in columns)


def _table_cell(value: Any, width: int, precision: int) -> str:
    if value is None:
        return " " * width
    if isinstance(value, int):
        return f"{value:>{width}d}"
    if isinstance(value, float):
        return f"{value:>{width}.{precision}f}"
    return f"{str(value):>{width}}"


def run_training(
    config: TrainingConfig,
    *,
    run_name: str | None,
    config_path: str,
) -> None:
    run_dir = _create_run_dir(
        output_dir=config.artifacts.output_dir,
        run_name=run_name,
        seed=config.train.seed,
    )
    checkpoints_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.csv"
    config_snapshot_path = run_dir / "config.json"
    tensorboard_dir = run_dir / "tensorboard"

    tracked_top_k = config.metrics_eval.top_k[0]
    tracked_reward_key = f"top_{tracked_top_k}_reward"
    tracked_diversity_key = f"top_{tracked_top_k}_diversity"
    tracked_utilization_key = f"top_{tracked_top_k}_utilization"

    run_config = config.to_snapshot()
    run_config.update(
        {
            "run_name": run_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
        }
    )
    _write_json(config_snapshot_path, run_config, indent=config.runtime.json_indent)

    metric_fieldnames = [
        "step",
        "elapsed_seconds",
        "train_steps_per_sec",
        "env_steps_per_sec",
        "loss",
        "mean_utilization",
        "mean_terminal_log_reward",
        "mean_log_pf",
        "mean_log_pb",
        "logZ",
        tracked_reward_key,
        tracked_diversity_key,
        tracked_utilization_key,
    ]

    selected_device = _resolve_device(config.runtime.device)
    if config.runtime.device == "gpu":
        print(
            f"Using device: {selected_device} "
            f"(requested={config.runtime.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, gpu_id={config.runtime.gpu_id}, XLA_FLAGS={os.environ.get('XLA_FLAGS')})"
        )
    else:
        print(f"Using device: {selected_device} (requested={config.runtime.device})")
    print(f"Artifacts directory: {run_dir}")

    with jax.default_device(selected_device):
        rng = jax.random.PRNGKey(config.train.seed)

        env = BinPackGFN(
            max_num_items=config.env.max_num_items,
            max_num_ems=config.env.max_num_ems,
            obs_num_ems=config.env.obs_num_ems,
            beta=config.env.beta,
            dense_reward=config.env.dense_reward,
        )
        rng_split = jax.random.split(rng, config.train.init_rng_split_count)
        rng = rng_split[0]
        env_init_key = rng_split[1]
        net_init_key = rng_split[2]
        eval_init_key = rng_split[3]
        env_params = env.init(env_init_key)

        obs_dim = env.observation_space["shape"][0]
        n_actions = env.action_space.n
        model = build_policy_transformer_from_config(
            obs_dim=obs_dim,
            num_actions=n_actions,
            obs_num_ems=config.env.obs_num_ems,
            max_num_items=config.env.max_num_items,
            key=net_init_key,
            model_config=config.model,
        )

        optimizer = optax.adam(config.train.learning_rate)
        params = (eqx.filter(model, eqx.is_array), jnp.asarray(config.train.logz_init, dtype=jnp.float32))
        opt_state = optimizer.init(params)

        train_state = TrainState(
            rng_key=rng,
            env=env,
            env_params=env_params,
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            logZ=jnp.asarray(config.train.logz_init, dtype=jnp.float32),
            num_envs=config.train.num_envs,
        )

        train_step = make_train_step(
            residual_power=config.loss.residual_power,
            rng_split_count=config.train.rng_split_count,
        )
        _, eval_policy_static = eqx.partition(model, eqx.is_array)
        eval_fwd_policy_fn = make_fwd_policy_fn(eval_policy_static, env.backward_action_space.n)
        metrics_module = gfnx.metrics.TopKMetricsModule(
            env=env,
            fwd_policy_fn=eval_fwd_policy_fn,
            num_traj=config.metrics_eval.num_traj,
            batch_size=config.metrics_eval.batch_size,
            top_k=config.metrics_eval.top_k,
            distance_fn=action_history_hamming_distance,
        )
        metrics_state = metrics_module.init(eval_init_key, metrics_module.InitArgs())

        @jax.jit
        def eval_policy_metrics(
            current_metrics_state: Any,
            rng_key: chex.PRNGKey,
            current_policy_params: Any,
            current_env_params: Any,
        ) -> tuple[Any, dict[str, chex.Array]]:
            next_metrics_state = metrics_module.process(
                current_metrics_state,
                rng_key,
                metrics_module.ProcessArgs(
                    policy_params=current_policy_params,
                    env_params=current_env_params,
                ),
            )
            return next_metrics_state, metrics_module.get(next_metrics_state)

        @jax.jit
        def eval_utilization_samples(
            rng_key: chex.PRNGKey,
            current_policy_params: Any,
            current_env_params: Any,
        ) -> chex.Array:
            _, rollout_info = gfnx.utils.forward_rollout(
                rng_key=rng_key,
                num_envs=config.metrics_eval.num_traj,
                policy_fn=eval_fwd_policy_fn,
                policy_params=current_policy_params,
                env=env,
                env_params=current_env_params,
            )
            return rollout_info["final_env_state"].volume_utilization

        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=metric_fieldnames)
            writer.writeheader()

            tb_writer = SummaryWriter(logdir=str(tensorboard_dir))
            start_time = time.perf_counter()
            prev_step_time = start_time
            table_columns = [
                ("step", 7),
                ("steps_s", 10),
                ("env_steps_s", 12),
                ("train_util", 10),
                (f"top_{tracked_top_k}_util", 12),
            ]
            table_header_printed = False
            try:
                for step in range(config.train.num_train_steps):
                    train_state, metrics = train_step(train_state)

                    now = time.perf_counter()
                    elapsed_seconds = max(now - start_time, 1e-9)
                    step_duration_seconds = max(now - prev_step_time, 1e-9)
                    prev_step_time = now
                    train_steps_per_sec = 1.0 / step_duration_seconds
                    env_steps_per_sec = config.train.num_envs / step_duration_seconds

                    top_k_reward: float | None = None
                    top_k_diversity: float | None = None
                    top_k_utilization: float | None = None
                    eval_utilization_samples_host: Any = None
                    should_log = (step % config.logging.every == 0) or (step == config.train.num_train_steps - 1)
                    if should_log:
                        eval_rng_key_metrics, eval_rng_key_util, next_rng_key = jax.random.split(train_state.rng_key, 3)
                        train_state = train_state._replace(rng_key=next_rng_key)
                        current_policy_params, _ = eqx.partition(train_state.model, eqx.is_array)
                        metrics_state, eval_results = eval_policy_metrics(
                            metrics_state,
                            eval_rng_key_metrics,
                            current_policy_params,
                            train_state.env_params,
                        )
                        top_k_reward = float(eval_results[tracked_reward_key])
                        top_k_diversity = float(eval_results[tracked_diversity_key])
                        eval_util_samples = eval_utilization_samples(
                            eval_rng_key_util,
                            current_policy_params,
                            train_state.env_params,
                        )
                        sorted_utilization = jnp.sort(eval_util_samples)
                        top_k_utilization = float(jnp.mean(sorted_utilization[-tracked_top_k:]))
                        eval_utilization_samples_host = jax.device_get(eval_util_samples)

                    row = {
                        "step": step,
                        "elapsed_seconds": elapsed_seconds,
                        "train_steps_per_sec": train_steps_per_sec,
                        "env_steps_per_sec": env_steps_per_sec,
                        "loss": float(metrics["loss"]),
                        "mean_utilization": float(metrics["mean_utilization"]),
                        "mean_terminal_log_reward": float(metrics["mean_terminal_log_reward"]),
                        "mean_log_pf": float(metrics["mean_log_pf"]),
                        "mean_log_pb": float(metrics["mean_log_pb"]),
                        "logZ": float(train_state.logZ),
                        tracked_reward_key: top_k_reward,
                        tracked_diversity_key: top_k_diversity,
                        tracked_utilization_key: top_k_utilization,
                    }
                    writer.writerow({key: _json_safe_value(value) for key, value in row.items()})

                    tb_writer.add_scalar("Loss/TB_Loss", row["loss"], step)
                    tb_writer.add_scalar("GFN/logZ", row["logZ"], step)
                    tb_writer.add_scalar("GFN/mean_log_pf", row["mean_log_pf"], step)
                    tb_writer.add_scalar("GFN/mean_log_pb", row["mean_log_pb"], step)
                    tb_writer.add_scalar("Performance/mean_utilization", row["mean_utilization"], step)
                    tb_writer.add_scalar("Performance/elapsed_seconds", row["elapsed_seconds"], step)
                    tb_writer.add_scalar("Performance/train_steps_per_sec", row["train_steps_per_sec"], step)
                    tb_writer.add_scalar("Performance/env_steps_per_sec", row["env_steps_per_sec"], step)
                    tb_writer.add_scalar(
                        "Performance/mean_terminal_log_reward",
                        row["mean_terminal_log_reward"],
                        step,
                    )
                    if should_log and top_k_reward is not None and top_k_diversity is not None and top_k_utilization is not None:
                        tb_writer.add_scalar(f"Eval/top_{tracked_top_k}_reward", top_k_reward, step)
                        tb_writer.add_scalar(f"Eval/top_{tracked_top_k}_diversity", top_k_diversity, step)
                        tb_writer.add_scalar(f"Eval/top_{tracked_top_k}_utilization", top_k_utilization, step)
                        if eval_utilization_samples_host is not None:
                            tb_writer.add_histogram("Eval/utilization_hist", eval_utilization_samples_host, step)

                    if should_log:
                        float_precision = 2
                        if not table_header_printed:
                            print(_format_train_log(_table_header(table_columns), config.runtime.train_log_color_code))
                            print(_format_train_log(_table_separator(table_columns), config.runtime.train_log_color_code))
                            table_header_printed = True

                        row_cells = [
                            _table_cell(step, 7, float_precision),
                            _table_cell(row["train_steps_per_sec"], 10, float_precision),
                            _table_cell(row["env_steps_per_sec"], 12, float_precision),
                            _table_cell(row["mean_utilization"], 10, float_precision),
                            _table_cell(top_k_utilization, 12, float_precision),
                        ]
                        table_row = " ".join(row_cells)
                        print(_format_train_log(table_row, config.runtime.train_log_color_code))

                    should_checkpoint = (
                        ((step + 1) % config.checkpointing.every == 0)
                        or (step == config.train.num_train_steps - 1)
                    )
                    if should_checkpoint:
                        width = config.checkpointing.filename_width
                        ckpt_name = f"step_{step + 1:0{width}d}.eqx"
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
                            indent=config.runtime.json_indent,
                        )
            finally:
                tb_writer.flush()
                tb_writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TB GFlowNet on Jumanji BinPack.")
    parser.add_argument(
        "--config",
        type=str,
        default=_BOOTSTRAP_CONFIG_PATH,
        help="Path to base YAML training config.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-num-items", type=int, default=None)
    parser.add_argument("--max-num-ems", type=int, default=None)
    parser.add_argument("--obs-num-ems", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
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
        default=None,
        help="Save checkpoints every N train steps and at final step.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Print training metrics every N train steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help="Execution device. Use 'gpu' to enable CUDA via JAX.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Physical GPU index to expose via CUDA_VISIBLE_DEVICES. Applied before importing jax.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = resolve_config(args.config, args)
    run_training(
        config,
        run_name=args.run_name,
        config_path=args.config,
    )
