from __future__ import annotations

import argparse
import importlib
from typing import Any, NamedTuple

import chex
import equinox as eqx
import gfnx
import jax
import jax.numpy as jnp
import optax

from env_wrapper import BinPackEnvParams, BinPackGFN


class PolicyMLP(eqx.Module):
    encoder: eqx.nn.MLP
    logits_head: eqx.nn.Linear
    flow_head: eqx.nn.Linear

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int, *, key: chex.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            key=k1,
        )
        self.logits_head = eqx.nn.Linear(hidden_dim, num_actions, key=k2)
        self.flow_head = eqx.nn.Linear(hidden_dim, 1, key=k3)

    def __call__(self, obs: chex.Array) -> tuple[chex.Array, chex.Array]:
        h = self.encoder(obs)
        logits = self.logits_head(h)
        log_flow = jnp.squeeze(self.flow_head(h), axis=-1)
        return logits, log_flow


class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    env: BinPackGFN
    env_params: BinPackEnvParams
    model: PolicyMLP
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    logZ: chex.Array
    num_envs: int


def _resolve_device(device_kind: str) -> jax.Device:
    if device_kind == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("No CPU device found by JAX.")
        return cpu_devices[0]

    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) <= 1:
        raise RuntimeError(
            "GPU device index 1 requested, but fewer than 2 GPU devices are available."
        )
    return gpu_devices[1]


def _resolve_tb_loss() -> Any:
    candidates = [
        ("gfnx.losses.trajectory_balance", "TrajectoryBalance"),
        ("gfnx.losses.tb", "TrajectoryBalance"),
        ("gfnx.loss.trajectory_balance", "TrajectoryBalance"),
    ]
    for module_name, symbol_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, symbol_name):
                return getattr(module, symbol_name)()
        except Exception:
            continue
    return None


def manual_tb_loss(
    model: PolicyMLP,
    logZ: chex.Array,
    traj_data: gfnx.utils.TrajectoryData,
    env: BinPackGFN,
    env_params: BinPackEnvParams,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    obs = traj_data.obs
    actions = traj_data.action
    done = traj_data.done
    pad = traj_data.pad
    states = traj_data.state

    batch_size, traj_len = actions.shape
    flat_obs = obs.reshape((batch_size * traj_len, -1))
    flat_states = jax.tree.map(
        lambda x: x.reshape((batch_size * traj_len,) + tuple(x.shape[2:])),
        states,
    )

    logits, _ = jax.vmap(model)(flat_obs)
    invalid_mask = env.get_invalid_mask(flat_states, env_params)
    masked_logits = gfnx.utils.mask_logits(logits, invalid_mask)
    log_pf_all = jax.nn.log_softmax(masked_logits, axis=-1)

    flat_actions = actions.reshape((-1, 1))
    log_pf_taken = jnp.take_along_axis(log_pf_all, flat_actions, axis=-1).squeeze(-1)
    log_pf_taken = log_pf_taken.reshape((batch_size, traj_len))

    valid_steps = jnp.logical_not(pad)
    log_pf_sum = jnp.sum(jnp.where(valid_steps, log_pf_taken, 0.0), axis=1)
    terminal_log_reward = jnp.sum(jnp.where(done, traj_data.log_gfn_reward, 0.0), axis=1)

    residual = logZ + log_pf_sum - terminal_log_reward
    loss = jnp.mean(residual**2)
    metrics = {
        "loss": loss,
        "mean_terminal_log_reward": jnp.mean(terminal_log_reward),
        "mean_log_pf": jnp.mean(log_pf_sum),
        "mean_utilization": jnp.mean(terminal_log_reward / env_params.reward_params.beta),
    }
    return loss, metrics


def make_train_step(tb_loss_module: Any):
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
            return logits, {}

        traj_data, _ = gfnx.utils.forward_rollout(
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
            if tb_loss_module is not None and hasattr(tb_loss_module, "loss"):
                try:
                    return tb_loss_module.loss(model, logZ, traj_data, train_state.env, rollout_env_params)
                except Exception:
                    pass
            return manual_tb_loss(model, logZ, traj_data, train_state.env, rollout_env_params)

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
) -> None:
    selected_device = _resolve_device(device)
    print(f"Using device: {selected_device} (requested={device})")

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
        model = PolicyMLP(obs_dim=obs_dim, num_actions=n_actions, hidden_dim=hidden_dim, key=net_init_key)

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

        tb_loss_module = _resolve_tb_loss()
        train_step = make_train_step(tb_loss_module)

        for step in range(num_train_steps):
            train_state, metrics = train_step(train_state)
            if step % 50 == 0 or step == num_train_steps - 1:
                print(
                    f"step={step:05d} "
                    f"loss={float(metrics['loss']):.6f} "
                    f"mean_utilization={float(metrics['mean_utilization']):.6f} "
                    f"mean_terminal_log_reward={float(metrics['mean_terminal_log_reward']):.6f} "
                    f"logZ={float(train_state.logZ):.6f}"
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
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
    )
