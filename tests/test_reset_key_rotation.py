from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from env_wrapper import BinPackGFN


class ResetKeyRotationTest(unittest.TestCase):
    def test_different_reset_keys_yield_different_initial_observations(self) -> None:
        env = BinPackGFN(
            max_num_items=6,
            max_num_ems=12,
            obs_num_ems=12,
            beta=10.0,
            dense_reward=False,
        )

        params_a = env.init(jax.random.PRNGKey(1))
        params_b = env.init(jax.random.PRNGKey(2))

        obs_a, _ = env.reset(num_envs=1, env_params=params_a)
        obs_b, _ = env.reset(num_envs=1, env_params=params_b)

        self.assertFalse(jnp.array_equal(obs_a, obs_b))


if __name__ == "__main__":
    unittest.main()
