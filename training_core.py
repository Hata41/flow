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


def _stepwise_trajectory_log_probs(
    env: BinPackGFN,
    traj_data: gfnx.utils.TrajectoryData,
    env_params: BinPackEnvParams,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    batch_size = traj_data.done.shape[0]

    def flatten_tree(tree: Any) -> Any:
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), tree)

    states = flatten_tree(jax.tree.map(lambda x: x[:, :-1], traj_data.state))
    next_states = flatten_tree(jax.tree.map(lambda x: x[:, 1:], traj_data.state))

    forward_logits = flatten_tree(traj_data.info["forward_logits"][:, :-1])
    backward_logits = flatten_tree(traj_data.info["backward_logits"][:, 1:])

    fwd_actions = flatten_tree(jax.tree.map(lambda x: x[:, :-1], traj_data.action))
    bwd_actions = env.get_backward_action(states, fwd_actions, next_states, env_params)

    fwd_action_mask = env.get_invalid_mask(states, env_params)
    bwd_action_mask = env.get_invalid_backward_mask(next_states, env_params)

    forward_logprobs = jax.nn.log_softmax(
        forward_logits,
        where=jnp.logical_not(fwd_action_mask),
        axis=-1,
    )
    sampled_forward_logprobs = jnp.take_along_axis(
        forward_logprobs,
        fwd_actions[..., None],
        axis=-1,
    ).squeeze(-1)

    backward_logprobs = jax.nn.log_softmax(
        backward_logits,
        where=jnp.logical_not(bwd_action_mask),
        axis=-1,
    )
    sampled_backward_logprobs = jnp.take_along_axis(
        backward_logprobs,
        bwd_actions[..., None],
        axis=-1,
    ).squeeze(-1)

    step_mask = jnp.logical_not(traj_data.pad[:, :-1])
    step_log_pf = sampled_forward_logprobs.reshape(batch_size, -1)
    step_log_pb = sampled_backward_logprobs.reshape(batch_size, -1)
    return step_log_pf, step_log_pb, step_mask


def sub_trajectory_balance_loss(
    model: Any,
    logZ: chex.Array,
    traj_data: gfnx.utils.TrajectoryData,
    rollout_info: dict[str, chex.Array],
    env: BinPackGFN,
    env_params: BinPackEnvParams,
    residual_power: int,
    subtb_lambda: float,
    subtb_length_weighting: bool,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    step_log_pf, step_log_pb, step_mask = _stepwise_trajectory_log_probs(env, traj_data, env_params)
    terminal_log_reward = rollout_info["log_gfn_reward"]

    state_log_flow = jax.vmap(lambda obs_seq: jax.vmap(lambda obs: model(obs)[1])(obs_seq))(traj_data.obs)
    state_log_flow = state_log_flow.at[:, 0].set(logZ)
    state_log_flow = jnp.where(traj_data.done, terminal_log_reward[:, None], state_log_flow)

    transition_delta = jnp.where(step_mask, step_log_pf - step_log_pb, 0.0)
    transition_prefix = jnp.concatenate(
        [
            jnp.zeros((transition_delta.shape[0], 1), dtype=transition_delta.dtype),
            jnp.cumsum(transition_delta, axis=-1),
        ],
        axis=-1,
    )

    state_mask = jnp.logical_not(traj_data.pad)
    num_states = state_log_flow.shape[1]
    idx = jnp.arange(num_states)
    subtraj_lengths = idx[None, :] - idx[:, None]
    upper_triangle = subtraj_lengths > 0
    pair_mask = state_mask[:, :, None] & state_mask[:, None, :] & upper_triangle[None, :, :]

    prefix_delta_matrix = transition_prefix[:, None, :] - transition_prefix[:, :, None]
    residual_matrix = state_log_flow[:, :, None] + prefix_delta_matrix - state_log_flow[:, None, :]

    base_weight_matrix = jnp.where(
        upper_triangle,
        jnp.power(subtb_lambda, jnp.maximum(subtraj_lengths - 1, 0)),
        0.0,
    )
    if subtb_length_weighting:
        length_scale = jnp.where(upper_triangle, subtraj_lengths, 0)
        base_weight_matrix = base_weight_matrix * length_scale

    weight_matrix = jnp.where(pair_mask, base_weight_matrix[None, :, :], 0.0)
    finite_pair_mask = pair_mask & jnp.isfinite(residual_matrix) & jnp.isfinite(weight_matrix)

    residual_powered = jnp.power(jnp.where(finite_pair_mask, residual_matrix, 0.0), residual_power)
    weighted_residual = jnp.where(finite_pair_mask, weight_matrix * residual_powered, 0.0)
    weight_sum = jnp.maximum(jnp.sum(jnp.where(finite_pair_mask, weight_matrix, 0.0)), 1e-8)
    loss = jnp.sum(weighted_residual) / weight_sum

    log_pf_traj = jnp.sum(jnp.where(step_mask, step_log_pf, 0.0), axis=-1)
    log_pb_traj = jnp.sum(jnp.where(step_mask, step_log_pb, 0.0), axis=-1)
    finite_mask = (
        jnp.isfinite(log_pf_traj)
        & jnp.isfinite(log_pb_traj)
        & jnp.isfinite(terminal_log_reward)
        & jnp.isfinite(loss)
    )
    safe_count = jnp.maximum(jnp.sum(finite_mask), 1)

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


def make_train_step(
    *,
    objective: str,
    residual_power: int,
    subtb_lambda: float,
    subtb_length_weighting: bool,
    rng_split_count: int,
):
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
            if objective == "subtb":
                return sub_trajectory_balance_loss(
                    model,
                    logZ,
                    traj_data,
                    rollout_info,
                    train_state.env,
                    rollout_env_params,
                    residual_power,
                    subtb_lambda,
                    subtb_length_weighting,
                )
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
