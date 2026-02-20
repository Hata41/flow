from __future__ import annotations

from typing import Any, NamedTuple

import chex
import equinox as eqx
import gfnx
import jax
import jax.numpy as jnp
import optax

from env_wrapper import BinPackEnvParams, BinPackGFN, GFNBinPackState


class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    env: BinPackGFN
    env_params: BinPackEnvParams
    model: Any
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    logZ: chex.Array
    num_envs: int


def trajectory_balance_loss(
    model: Any,
    logZ: chex.Array,
    traj_data: gfnx.utils.TrajectoryData,
    rollout_info: dict[str, chex.Array],
    env: BinPackGFN,
    env_params: BinPackEnvParams,
    residual_power: int,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    del model
    log_pf_traj, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(env, traj_data, env_params)
    terminal_log_reward = rollout_info["log_gfn_reward"]

    residual = logZ + log_pf_traj - log_pb_traj - terminal_log_reward
    finite_mask = (
        jnp.isfinite(log_pf_traj)
        & jnp.isfinite(log_pb_traj)
        & jnp.isfinite(terminal_log_reward)
        & jnp.isfinite(residual)
    )
    safe_count = jnp.maximum(jnp.sum(finite_mask), 1)
    residual_powered = jnp.power(jnp.where(finite_mask, residual, 0.0), residual_power)
    loss = jnp.sum(residual_powered) / safe_count

    def _masked_mean(values: chex.Array) -> chex.Array:
        return jnp.sum(jnp.where(finite_mask, values, 0.0)) / safe_count

    metrics = {
        "loss": loss,
        "mean_terminal_log_reward": _masked_mean(terminal_log_reward),
        "mean_log_pf": _masked_mean(log_pf_traj),
        "mean_log_pb": _masked_mean(log_pb_traj),
        "mean_utilization": _masked_mean(terminal_log_reward / env_params.reward_params.beta),
    }
    return loss, metrics


def make_fwd_policy_fn(policy_static: Any, backward_action_dim: int):
    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey,
        env_obs: chex.Array,
        policy_params: Any,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del fwd_rng_key
        model = eqx.combine(policy_params, policy_static)
        logits, _ = jax.vmap(model)(env_obs)
        backward_logits = jnp.zeros((logits.shape[0], backward_action_dim), dtype=logits.dtype)
        return logits, {
            "forward_logits": logits,
            "backward_logits": backward_logits,
        }

    return fwd_policy_fn


def action_history_hamming_distance(lhs_state: GFNBinPackState, rhs_state: GFNBinPackState) -> chex.Array:
    return jnp.mean(lhs_state.action_history != rhs_state.action_history)


def make_train_step(*, residual_power: int, rng_split_count: int):
    @eqx.filter_jit
    def train_step(train_state: TrainState) -> tuple[TrainState, dict[str, chex.Array]]:
        split_keys = jax.random.split(train_state.rng_key, rng_split_count)
        rng_key = split_keys[0]
        rollout_key = split_keys[1]
        new_env_key = split_keys[2]

        rollout_env_params = eqx.tree_at(
            lambda p: p.reset_key,
            train_state.env_params,
            new_env_key,
        )

        policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
        fwd_policy_fn = make_fwd_policy_fn(policy_static, train_state.env.backward_action_space.n)

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
                residual_power,
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
