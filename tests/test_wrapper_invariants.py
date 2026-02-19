from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from env_wrapper import BinPackGFN


class WrapperInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = BinPackGFN(
            max_num_items=6,
            max_num_ems=12,
            obs_num_ems=12,
            beta=10.0,
            dense_reward=False,
        )
        self.params = self.env.init(jax.random.PRNGKey(0))

    def _first_valid_forward_action(self, state) -> jnp.ndarray:
        valid_flat = state.jumanji_state.action_mask.reshape(-1)
        self.assertTrue(bool(jnp.any(valid_flat)))
        return jnp.argmax(valid_flat.astype(jnp.int32)).astype(jnp.int32)

    def test_flatten_unflatten_is_bijective(self) -> None:
        flat_actions = jnp.arange(self.env.action_space.n, dtype=jnp.int32)

        def _round_trip(action: jnp.ndarray) -> jnp.ndarray:
            pair = self.env._unflatten_action(action)
            return self.env._flatten_action(pair[0], pair[1])

        round_trip = jax.vmap(_round_trip)(flat_actions)
        self.assertTrue(bool(jnp.array_equal(round_trip, flat_actions)))

    def test_forward_invalid_mask_matches_inverted_jumanji_mask(self) -> None:
        _, state = self.env.reset(num_envs=1, env_params=self.params)
        invalid_mask = self.env.get_invalid_mask(state, self.params)
        jumanji_valid_mask = state.jumanji_state.action_mask.reshape((1, -1))
        self.assertTrue(bool(jnp.array_equal(invalid_mask, jnp.logical_not(jumanji_valid_mask))))

    def test_backward_mask_is_deterministic_lifo_validity(self) -> None:
        _, state = self.env.reset(num_envs=1, env_params=self.params)
        invalid_backward_initial = self.env.get_invalid_backward_mask(state, self.params)
        self.assertTrue(bool(invalid_backward_initial[0, 0]))

        action = self._first_valid_forward_action(jax.tree.map(lambda x: x[0], state))
        _, next_state, _, _, _ = self.env.step(state, action[None], self.params)
        invalid_backward_after_step = self.env.get_invalid_backward_mask(next_state, self.params)
        self.assertFalse(bool(invalid_backward_after_step[0, 0]))

    def test_backward_resimulation_restores_previous_state(self) -> None:
        _, state0 = self.env.reset(num_envs=1, env_params=self.params)
        action0 = self._first_valid_forward_action(jax.tree.map(lambda x: x[0], state0))
        _, state1, _, done1, _ = self.env.step(state0, action0[None], self.params)
        self.assertFalse(bool(done1[0]))

        action1 = self._first_valid_forward_action(jax.tree.map(lambda x: x[0], state1))
        _, state2, _, _, _ = self.env.step(state1, action1[None], self.params)

        backward_action = jnp.zeros((1,), dtype=jnp.int32)
        _, prev_state, _, _, _ = self.env.backward_step(state2, backward_action, self.params)

        self.assertTrue(bool(jnp.array_equal(prev_state.step_count, state1.step_count)))
        self.assertTrue(bool(jnp.array_equal(prev_state.action_history, state1.action_history)))
        self.assertTrue(bool(jnp.allclose(prev_state.volume_utilization, state1.volume_utilization)))

        same_jumanji_state = jax.tree_util.tree_all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b),
                prev_state.jumanji_state,
                state1.jumanji_state,
            )
        )
        self.assertTrue(bool(same_jumanji_state))


if __name__ == "__main__":
    unittest.main()
