from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    gpu_id: int
    default_gpu_env_fallback: int
    train_log_color_code: int
    eval_log_color_code: int
    json_indent: int


@dataclass(frozen=True)
class EnvConfig:
    beta: float
    max_num_items: int
    max_num_ems: int
    obs_num_ems: int
    dense_reward: bool


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_multiplier: int
    qk_size_min: int
    obs_ems_feature_factor: int
    obs_item_feature_factor: int
    ems_coord_dim: int
    item_feature_dim: int
    mask_threshold: float
    key_count_base: int
    key_count_per_layer: int
    key_offset_initial: int
    flow_input_multiplier: int
    flow_output_dim: int
    policy_ems_head_key_index: int
    policy_item_head_key_index: int
    flow_head_key_index: int


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    num_train_steps: int
    num_envs: int
    learning_rate: float
    logz_init: float
    rng_split_count: int
    init_rng_split_count: int


@dataclass(frozen=True)
class LossConfig:
    objective: str
    residual_power: int
    subtb_lambda: float
    subtb_length_weighting: bool


@dataclass(frozen=True)
class MetricsEvalConfig:
    num_traj: int
    batch_size: int
    top_k: list[int]
    reward_epsilon: float


@dataclass(frozen=True)
class CheckpointConfig:
    every: int
    filename_width: int


@dataclass(frozen=True)
class LoggingConfig:
    every: int
    step_width: int
    float_precision: int


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    runtime: RuntimeConfig
    env: EnvConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
    metrics_eval: MetricsEvalConfig
    checkpointing: CheckpointConfig
    logging: LoggingConfig
    artifacts: ArtifactConfig

    def to_snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "seed": self.train.seed,
                "num_train_steps": self.train.num_train_steps,
                "num_envs": self.train.num_envs,
                "learning_rate": self.train.learning_rate,
                "hidden_dim": self.model.hidden_dim,
                "beta": self.env.beta,
                "max_num_items": self.env.max_num_items,
                "max_num_ems": self.env.max_num_ems,
                "obs_num_ems": self.env.obs_num_ems,
                "device": self.runtime.device,
                "gpu_id": self.runtime.gpu_id,
                "output_dir": self.artifacts.output_dir,
                "checkpoint_every": self.checkpointing.every,
                "log_every": self.logging.every,
            }
        )
        return payload


def _workspace_default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.base.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def _ensure_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid section '{name}' in config.")
    return section


def _coerce_top_k(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        raise ValueError("metrics_eval.top_k must be a list of ints.")
    return [int(value) for value in raw]


def _to_training_config(raw: dict[str, Any]) -> TrainingConfig:
    runtime = _ensure_section(raw, "runtime")
    env = _ensure_section(raw, "env")
    model = _ensure_section(raw, "model")
    train = _ensure_section(raw, "train")
    loss = _ensure_section(raw, "loss")
    metrics_eval = _ensure_section(raw, "metrics_eval")
    checkpointing = _ensure_section(raw, "checkpointing")
    logging = _ensure_section(raw, "logging")
    artifacts = _ensure_section(raw, "artifacts")

    config = TrainingConfig(
        runtime=RuntimeConfig(
            device=str(runtime["device"]),
            gpu_id=int(runtime["gpu_id"]),
            default_gpu_env_fallback=int(runtime["default_gpu_env_fallback"]),
            train_log_color_code=int(runtime["train_log_color_code"]),
            eval_log_color_code=int(runtime["eval_log_color_code"]),
            json_indent=int(runtime["json_indent"]),
        ),
        env=EnvConfig(
            beta=float(env["beta"]),
            max_num_items=int(env["max_num_items"]),
            max_num_ems=int(env["max_num_ems"]),
            obs_num_ems=int(env["obs_num_ems"]),
            dense_reward=bool(env["dense_reward"]),
        ),
        model=ModelConfig(
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            ff_multiplier=int(model["ff_multiplier"]),
            qk_size_min=int(model["qk_size_min"]),
            obs_ems_feature_factor=int(model["obs_ems_feature_factor"]),
            obs_item_feature_factor=int(model["obs_item_feature_factor"]),
            ems_coord_dim=int(model["ems_coord_dim"]),
            item_feature_dim=int(model["item_feature_dim"]),
            mask_threshold=float(model["mask_threshold"]),
            key_count_base=int(model["key_count_base"]),
            key_count_per_layer=int(model["key_count_per_layer"]),
            key_offset_initial=int(model["key_offset_initial"]),
            flow_input_multiplier=int(model["flow_input_multiplier"]),
            flow_output_dim=int(model["flow_output_dim"]),
            policy_ems_head_key_index=int(model["policy_ems_head_key_index"]),
            policy_item_head_key_index=int(model["policy_item_head_key_index"]),
            flow_head_key_index=int(model["flow_head_key_index"]),
        ),
        train=TrainConfig(
            seed=int(train["seed"]),
            num_train_steps=int(train["num_train_steps"]),
            num_envs=int(train["num_envs"]),
            learning_rate=float(train["learning_rate"]),
            logz_init=float(train["logz_init"]),
            rng_split_count=int(train["rng_split_count"]),
            init_rng_split_count=int(train["init_rng_split_count"]),
        ),
        loss=LossConfig(
            objective=str(loss.get("objective", "tb")).lower(),
            residual_power=int(loss["residual_power"]),
            subtb_lambda=float(loss.get("subtb_lambda", 0.9)),
            subtb_length_weighting=bool(loss.get("subtb_length_weighting", False)),
        ),
        metrics_eval=MetricsEvalConfig(
            num_traj=int(metrics_eval["num_traj"]),
            batch_size=int(metrics_eval["batch_size"]),
            top_k=_coerce_top_k(metrics_eval["top_k"]),
            reward_epsilon=float(metrics_eval["reward_epsilon"]),
        ),
        checkpointing=CheckpointConfig(
            every=int(checkpointing["every"]),
            filename_width=int(checkpointing["filename_width"]),
        ),
        logging=LoggingConfig(
            every=int(logging["every"]),
            step_width=int(logging["step_width"]),
            float_precision=int(logging["float_precision"]),
        ),
        artifacts=ArtifactConfig(
            output_dir=str(artifacts["output_dir"]),
        ),
    )
    validate_config(config)
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must deserialize into a mapping: {path}")
    return payload


def load_config(config_path: str | None) -> TrainingConfig:
    path = Path(config_path) if config_path else _workspace_default_config_path()
    raw = _load_yaml(path)
    return _to_training_config(raw)


def _apply_cli_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, dict[str, Any]] = {
        "runtime": {},
        "env": {},
        "model": {},
        "train": {},
        "checkpointing": {},
        "logging": {},
        "artifacts": {},
    }

    def set_if_present(section: str, field: str, value: Any) -> None:
        if value is not None:
            overrides[section][field] = value

    set_if_present("runtime", "device", getattr(args, "device", None))
    set_if_present("runtime", "gpu_id", getattr(args, "gpu_id", None))

    set_if_present("train", "seed", getattr(args, "seed", None))
    set_if_present("train", "num_train_steps", getattr(args, "num_train_steps", None))
    set_if_present("train", "num_envs", getattr(args, "num_envs", None))
    set_if_present("train", "learning_rate", getattr(args, "learning_rate", None))

    set_if_present("model", "hidden_dim", getattr(args, "hidden_dim", None))
    set_if_present("env", "beta", getattr(args, "beta", None))
    set_if_present("env", "max_num_items", getattr(args, "max_num_items", None))
    set_if_present("env", "max_num_ems", getattr(args, "max_num_ems", None))
    set_if_present("env", "obs_num_ems", getattr(args, "obs_num_ems", None))

    set_if_present("artifacts", "output_dir", getattr(args, "output_dir", None))
    set_if_present("checkpointing", "every", getattr(args, "checkpoint_every", None))
    set_if_present("logging", "every", getattr(args, "log_every", None))

    compact_override = {key: value for key, value in overrides.items() if value}
    return _deep_merge(raw, compact_override)


def resolve_config(config_path: str | None, args: argparse.Namespace) -> TrainingConfig:
    path = Path(config_path) if config_path else _workspace_default_config_path()
    raw = _load_yaml(path)
    merged = _apply_cli_overrides(raw, args)
    return _to_training_config(merged)


def bootstrap_runtime_from_argv(argv: list[str]) -> tuple[str, int, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    args, _ = parser.parse_known_args(argv)

    cfg = load_config(args.config)
    device = args.device if args.device is not None else cfg.runtime.device
    gpu_id = args.gpu_id if args.gpu_id is not None else cfg.runtime.gpu_id

    if device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        use_safe_gpu_xla = os.environ.get("FLOW_SAFE_GPU_XLA", "1") != "0"
        if use_safe_gpu_xla:
            existing_xla_flags = os.environ.get("XLA_FLAGS", "")
            fallback_xla_flags = [
                "--xla_gpu_enable_triton_gemm=false",
                "--xla_gpu_autotune_level=0",
            ]
            merged_flags = [existing_xla_flags] if existing_xla_flags else []
            for flag in fallback_xla_flags:
                if flag not in existing_xla_flags:
                    merged_flags.append(flag)
            if merged_flags:
                os.environ["XLA_FLAGS"] = " ".join(merged_flags)

    return device, gpu_id, str(Path(args.config) if args.config else _workspace_default_config_path())


def _require_positive_int(value: int, field: str) -> None:
    if value <= 0:
        raise ValueError(f"{field} must be > 0 (got {value}).")


def validate_config(config: TrainingConfig) -> None:
    if config.runtime.device not in {"cpu", "gpu"}:
        raise ValueError(f"runtime.device must be cpu or gpu (got {config.runtime.device}).")

    _require_positive_int(config.runtime.json_indent, "runtime.json_indent")
    _require_positive_int(config.train.num_train_steps, "train.num_train_steps")
    _require_positive_int(config.train.num_envs, "train.num_envs")
    _require_positive_int(config.train.rng_split_count, "train.rng_split_count")
    _require_positive_int(config.train.init_rng_split_count, "train.init_rng_split_count")
    _require_positive_int(config.env.max_num_items, "env.max_num_items")
    _require_positive_int(config.env.max_num_ems, "env.max_num_ems")
    _require_positive_int(config.env.obs_num_ems, "env.obs_num_ems")
    _require_positive_int(config.model.hidden_dim, "model.hidden_dim")
    _require_positive_int(config.model.num_layers, "model.num_layers")
    _require_positive_int(config.model.num_heads, "model.num_heads")
    _require_positive_int(config.model.ff_multiplier, "model.ff_multiplier")
    _require_positive_int(config.model.qk_size_min, "model.qk_size_min")
    _require_positive_int(config.model.obs_ems_feature_factor, "model.obs_ems_feature_factor")
    _require_positive_int(config.model.obs_item_feature_factor, "model.obs_item_feature_factor")
    _require_positive_int(config.model.ems_coord_dim, "model.ems_coord_dim")
    _require_positive_int(config.model.item_feature_dim, "model.item_feature_dim")
    _require_positive_int(config.model.key_count_base, "model.key_count_base")
    _require_positive_int(config.model.key_count_per_layer, "model.key_count_per_layer")
    _require_positive_int(config.model.flow_input_multiplier, "model.flow_input_multiplier")
    _require_positive_int(config.model.flow_output_dim, "model.flow_output_dim")
    _require_positive_int(config.checkpointing.every, "checkpointing.every")
    _require_positive_int(config.checkpointing.filename_width, "checkpointing.filename_width")
    _require_positive_int(config.logging.every, "logging.every")
    _require_positive_int(config.logging.step_width, "logging.step_width")
    _require_positive_int(config.logging.float_precision, "logging.float_precision")
    _require_positive_int(config.metrics_eval.num_traj, "metrics_eval.num_traj")
    _require_positive_int(config.metrics_eval.batch_size, "metrics_eval.batch_size")
    _require_positive_int(config.loss.residual_power, "loss.residual_power")

    if config.train.learning_rate <= 0:
        raise ValueError("train.learning_rate must be > 0.")
    if config.loss.objective not in {"tb", "subtb"}:
        raise ValueError(f"loss.objective must be one of {{'tb', 'subtb'}} (got {config.loss.objective}).")
    if not (0.0 < config.loss.subtb_lambda <= 1.0):
        raise ValueError("loss.subtb_lambda must satisfy 0 < subtb_lambda <= 1.")
    if config.env.beta <= 0:
        raise ValueError("env.beta must be > 0.")
    if config.metrics_eval.reward_epsilon <= 0:
        raise ValueError("metrics_eval.reward_epsilon must be > 0.")
    if config.model.mask_threshold < 0:
        raise ValueError("model.mask_threshold must be >= 0.")
    if config.model.hidden_dim < config.model.num_heads:
        raise ValueError("model.hidden_dim must be >= model.num_heads.")
    if config.metrics_eval.batch_size > config.metrics_eval.num_traj:
        raise ValueError("metrics_eval.batch_size must be <= metrics_eval.num_traj.")

    if len(config.metrics_eval.top_k) == 0:
        raise ValueError("metrics_eval.top_k must contain at least one value.")
    for top_k in config.metrics_eval.top_k:
        _require_positive_int(top_k, "metrics_eval.top_k[]")
        if top_k > config.metrics_eval.num_traj:
            raise ValueError(
                f"metrics_eval.top_k value {top_k} cannot exceed metrics_eval.num_traj={config.metrics_eval.num_traj}."
            )

    expected_key_count = config.model.key_count_base + (
        config.model.key_count_per_layer * config.model.num_layers
    )
    if expected_key_count <= 0:
        raise ValueError("model key count must be positive.")
