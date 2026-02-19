from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from gfnx.base import BaseEnvParams, BaseEnvState, BaseRewardModule, BaseVecEnvironment
from jumanji.environments.packing.bin_pack.env import BinPack
from jumanji.environments.packing.bin_pack.generator import RandomGenerator
from jumanji.environments.packing.bin_pack.reward import DenseReward, SparseReward
from jumanji.environments.packing.bin_pack.types import State as JumanjiBinPackState


@dataclass(frozen=True)
class DiscreteSpace:
    n: int


@chex.dataclass(frozen=True)
class BinPackRewardParams:
    beta: chex.Array
    epsilon: chex.Array


@chex.dataclass(frozen=True)
class BinPackEnvParams(BaseEnvParams):
    reward_params: BinPackRewardParams
    reset_key: chex.PRNGKey


@chex.dataclass(frozen=True)
class GFNBinPackState(BaseEnvState):
    jumanji_state: JumanjiBinPackState
    step_count: chex.Array
    action_history: chex.Array
    init_key: chex.PRNGKey
    volume_utilization: chex.Array


class BinPackRewardModule(BaseRewardModule[GFNBinPackState, BinPackEnvParams]):
    def __init__(self, beta: float = 10.0, epsilon: float = 1e-8):
        self._default_beta = jnp.asarray(beta, dtype=jnp.float32)
        self._default_epsilon = jnp.asarray(epsilon, dtype=jnp.float32)

    def init(self, rng_key: chex.PRNGKey, dummy_state: GFNBinPackState) -> BinPackRewardParams:
        del rng_key, dummy_state
        return BinPackRewardParams(beta=self._default_beta, epsilon=self._default_epsilon)

    def log_reward(
        self, state: GFNBinPackState, env_params: BinPackEnvParams
    ) -> chex.Array:
        beta = env_params.reward_params.beta
        return beta * state.volume_utilization

    def reward(
        self, state: GFNBinPackState, env_params: BinPackEnvParams
    ) -> chex.Array:
        return jnp.exp(self.log_reward(state, env_params))


class BinPackGFN(BaseVecEnvironment[GFNBinPackState, BinPackEnvParams]):
    def __init__(
        self,
        max_num_items: int = 20,
        max_num_ems: int = 40,
        obs_num_ems: int = 40,
        normalize_dimensions: bool = True,
        dense_reward: bool = False,
        beta: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        reward_module = BinPackRewardModule(beta=beta, epsilon=epsilon)
        super().__init__(reward_module=reward_module)

        self.max_num_items = max_num_items
        self.max_num_ems = max_num_ems
        self.obs_num_ems = obs_num_ems
        self._n_actions = obs_num_ems * max_num_items
        self._obs_dim = 7 * obs_num_ems + 5 * max_num_items + self._n_actions
        self._action_space = DiscreteSpace(n=self._n_actions)
        self._backward_action_space = DiscreteSpace(n=1)
        self._observation_space: Dict[str, Any] = {"shape": (self._obs_dim,), "dtype": jnp.float32}
        self._state_space: Dict[str, Any] = {
            "step_count": ((), jnp.int32),
            "action_history": ((max_num_items, 2), jnp.int32),
        }

        generator = RandomGenerator(max_num_items=max_num_items, max_num_ems=max_num_ems)
        reward_fn = DenseReward() if dense_reward else SparseReward()
        self.jumanji_env = BinPack(
            generator=generator,
            obs_num_ems=obs_num_ems,
            reward_fn=reward_fn,
            normalize_dimensions=normalize_dimensions,
        )

    @property
    def name(self) -> str:
        return "BinPackGFN-v0"

    @property
    def max_steps_in_episode(self) -> int:
        return self.max_num_items

    @property
    def action_space(self) -> DiscreteSpace:
        return self._action_space

    @property
    def backward_action_space(self) -> DiscreteSpace:
        return self._backward_action_space

    @property
    def observation_space(self) -> Dict[str, Any]:
        return self._observation_space

    @property
    def state_space(self) -> Dict[str, Any]:
        return self._state_space

    def init(self, rng_key: chex.PRNGKey) -> BinPackEnvParams:
        dummy_state = self._single_init_state(rng_key)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return BinPackEnvParams(reward_params=reward_params, reset_key=rng_key)

    def reset(self, num_envs: int, env_params: BinPackEnvParams) -> tuple[chex.ArrayTree, GFNBinPackState]:
        keys = jax.random.split(env_params.reset_key, num_envs)
        state = jax.vmap(self._single_init_state)(keys)
        return self.get_obs(state, env_params), state

    def get_init_state(self, num_envs: int) -> GFNBinPackState:
        del num_envs
        raise NotImplementedError(
            "BinPackGFN uses stateless reset(env_params.reset_key); call reset(...) directly."
        )

    def _single_init_state(self, key: chex.PRNGKey) -> GFNBinPackState:
        jumanji_state, timestep = self.jumanji_env.reset(key)
        history = jnp.full((self.max_num_items, 2), -1, dtype=jnp.int32)
        utilization = self._extract_volume_utilization(timestep)
        return GFNBinPackState(
            jumanji_state=jumanji_state,
            is_terminal=jnp.asarray(timestep.last(), dtype=jnp.bool_),
            is_initial=jnp.asarray(True, dtype=jnp.bool_),
            is_pad=jnp.asarray(False, dtype=jnp.bool_),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            action_history=history,
            init_key=key,
            volume_utilization=utilization,
        )

    def _extract_volume_utilization(self, timestep: Any) -> chex.Array:
        extras = getattr(timestep, "extras", None)
        if extras is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        if hasattr(extras, "get"):
            return jnp.asarray(extras.get("volume_utilization", 0.0), dtype=jnp.float32)
        return jnp.asarray(0.0, dtype=jnp.float32)

    def _flatten_action(self, ems_id: chex.Array, item_id: chex.Array) -> chex.Array:
        return ems_id * self.max_num_items + item_id

    def _unflatten_action(self, action: chex.Array) -> chex.Array:
        ems_id = action // self.max_num_items
        item_id = action % self.max_num_items
        return jnp.stack([ems_id, item_id], axis=0).astype(jnp.int32)

    def _single_observation(self, state: GFNBinPackState) -> chex.Array:
        jstate = state.jumanji_state
        obs_ems_indices = jstate.sorted_ems_indexes[: self.obs_num_ems]

        ems_coords = jnp.stack(
            [
                jstate.ems.x1[obs_ems_indices],
                jstate.ems.x2[obs_ems_indices],
                jstate.ems.y1[obs_ems_indices],
                jstate.ems.y2[obs_ems_indices],
                jstate.ems.z1[obs_ems_indices],
                jstate.ems.z2[obs_ems_indices],
            ],
            axis=-1,
        ).astype(jnp.float32)
        ems_mask = jstate.ems_mask[obs_ems_indices]
        ems_coords = ems_coords * ems_mask[:, None].astype(jnp.float32)

        item_feats = jnp.stack(
            [
                jstate.items.x_len,
                jstate.items.y_len,
                jstate.items.z_len,
            ],
            axis=-1,
        ).astype(jnp.float32)

        obs = jnp.concatenate(
            [
                ems_coords.reshape(-1),
                ems_mask.astype(jnp.float32),
                item_feats.reshape(-1),
                jstate.items_mask.astype(jnp.float32),
                jstate.items_placed.astype(jnp.float32),
                jstate.action_mask[: self.obs_num_ems].astype(jnp.float32).reshape(-1),
            ],
            axis=0,
        )
        chex.assert_shape(obs, (self._obs_dim,))
        return obs

    def get_obs(
        self, state: GFNBinPackState, env_params: BinPackEnvParams
    ) -> chex.Array:
        del env_params
        return jax.vmap(self._single_observation)(state)

    def _single_transition(
        self,
        state: GFNBinPackState,
        action: chex.Array,
        env_params: BinPackEnvParams,
    ) -> Tuple[GFNBinPackState, chex.Array, Dict[str, Any]]:
        del env_params
        action_pair = self._unflatten_action(action)
        next_jstate, timestep = self.jumanji_env.step(state.jumanji_state, action_pair)

        history = jax.lax.cond(
            state.step_count < self.max_num_items,
            lambda h: h.at[state.step_count].set(action_pair),
            lambda h: h,
            state.action_history,
        )
        next_step_count = jnp.minimum(state.step_count + 1, self.max_num_items)
        utilization = self._extract_volume_utilization(timestep)
        next_state = GFNBinPackState(
            jumanji_state=next_jstate,
            is_terminal=jnp.asarray(timestep.last(), dtype=jnp.bool_),
            is_initial=jnp.asarray(False, dtype=jnp.bool_),
            is_pad=state.is_pad,
            step_count=next_step_count,
            action_history=history,
            init_key=state.init_key,
            volume_utilization=utilization,
        )
        done = jnp.asarray(timestep.last(), dtype=jnp.bool_)
        info = {"volume_utilization": utilization}
        return next_state, done, info

    def _single_backward_transition(
        self,
        state: GFNBinPackState,
        backward_action: chex.Array,
        env_params: BinPackEnvParams,
    ) -> Tuple[GFNBinPackState, chex.Array, Dict[str, Any]]:
        del backward_action, env_params

        has_previous_step = state.step_count > 0
        target_steps = jnp.maximum(state.step_count - 1, 0)

        def _resimulate(_: None) -> GFNBinPackState:
            jstate, timestep = self.jumanji_env.reset(state.init_key)

            def _scan_body(
                carry: Tuple[JumanjiBinPackState, Any],
                idx: chex.Array,
            ) -> Tuple[Tuple[JumanjiBinPackState, Any], None]:
                current_jstate, current_timestep = carry
                action_pair = state.action_history[idx]

                def _do_step(__: None) -> Tuple[JumanjiBinPackState, Any]:
                    next_jstate, next_timestep = self.jumanji_env.step(current_jstate, action_pair)
                    return next_jstate, next_timestep

                next_carry = jax.lax.cond(
                    idx < target_steps,
                    _do_step,
                    lambda __: (current_jstate, current_timestep),
                    operand=None,
                )
                return next_carry, None

            (rebuilt_jstate, rebuilt_timestep), _ = jax.lax.scan(
                _scan_body,
                (jstate, timestep),
                jnp.arange(self.max_num_items, dtype=jnp.int32),
            )

            cleaned_history = state.action_history.at[target_steps].set(
                jnp.array([-1, -1], dtype=jnp.int32)
            )
            utilization = self._extract_volume_utilization(rebuilt_timestep)
            return GFNBinPackState(
                jumanji_state=rebuilt_jstate,
                is_terminal=jnp.asarray(rebuilt_timestep.last(), dtype=jnp.bool_),
                is_initial=jnp.asarray(target_steps == 0, dtype=jnp.bool_),
                is_pad=state.is_pad,
                step_count=target_steps,
                action_history=cleaned_history,
                init_key=state.init_key,
                volume_utilization=utilization,
            )

        prev_state = jax.lax.cond(
            has_previous_step,
            _resimulate,
            lambda _: state,
            operand=None,
        )
        done = jnp.asarray(prev_state.is_initial, dtype=jnp.bool_)
        info = {"backward_valid": has_previous_step}
        return prev_state, done, info

    def get_backward_action(
        self,
        state: GFNBinPackState,
        forward_action: chex.Array,
        next_state: GFNBinPackState,
        env_params: BinPackEnvParams,
    ) -> chex.Array:
        del state, next_state, env_params
        return jnp.zeros((forward_action.shape[0],), dtype=jnp.int32)

    def get_forward_action(
        self,
        state: GFNBinPackState,
        backward_action: chex.Array,
        prev_state: GFNBinPackState,
        env_params: BinPackEnvParams,
    ) -> chex.Array:
        del backward_action, env_params

        def _single(
            child_state: GFNBinPackState,
            parent_state: GFNBinPackState,
        ) -> chex.Array:
            idx = jnp.clip(parent_state.step_count, 0, self.max_num_items - 1)
            pair = child_state.action_history[idx]
            return self._flatten_action(pair[0], pair[1]).astype(jnp.int32)

        return jax.vmap(_single)(state, prev_state)

    def get_invalid_mask(
        self,
        state: GFNBinPackState,
        env_params: BinPackEnvParams,
    ) -> chex.Array:
        del env_params

        def _single_invalid(single_state: GFNBinPackState) -> chex.Array:
            valid_flat = single_state.jumanji_state.action_mask.reshape(-1)
            invalid_flat = jnp.logical_not(valid_flat)
            chex.assert_shape(invalid_flat, (self._n_actions,))
            return invalid_flat

        return jax.vmap(_single_invalid)(state)

    def get_invalid_backward_mask(
        self,
        state: GFNBinPackState,
        env_params: BinPackEnvParams,
    ) -> chex.Array:
        del env_params

        def _single_invalid_backward(single_state: GFNBinPackState) -> chex.Array:
            mask = jnp.ones((self._backward_action_space.n,), dtype=jnp.bool_)
            return mask.at[0].set(single_state.step_count == 0)

        return jax.vmap(_single_invalid_backward)(state)
