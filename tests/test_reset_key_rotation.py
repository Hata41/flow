from __future__ import annotations

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from env_wrapper import BinPackGFN
from train import PolicyMLP, TrainState, make_train_step


class ResetKeyRotationTest(unittest.TestCase):
    def test_env_reset_key_is_rotated_in_train_step(self) -> None:
        seed = 0
        rng = jax.random.PRNGKey(seed)

        env = BinPackGFN(
            max_num_items=10,
            max_num_ems=30,
            obs_num_ems=30,
            beta=10.0,
            dense_reward=False,
        )

        rng, env_init_key, net_init_key = jax.random.split(rng, 3)
        env_params = env.init(env_init_key)

        model = PolicyMLP(
            obs_dim=env.observation_space["shape"][0],
            num_actions=env.action_space.n,
            hidden_dim=32,
            key=net_init_key,
        )

        optimizer = optax.adam(1e-3)
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
            num_envs=8,
        )

        train_step = make_train_step(tb_loss_module=None)

        state_1, _ = train_step(train_state)
        state_2, _ = train_step(state_1)

        self.assertFalse(jnp.array_equal(train_state.env_params.reset_key, state_1.env_params.reset_key))
        self.assertFalse(jnp.array_equal(state_1.env_params.reset_key, state_2.env_params.reset_key))

    def test_different_reset_keys_yield_different_initial_observations(self) -> None:
        env = BinPackGFN(
            max_num_items=10,
            max_num_ems=30,
            obs_num_ems=30,
            beta=10.0,
            dense_reward=False,
        )

        params_a = env.init(jax.random.PRNGKey(1))
        params_b = env.init(jax.random.PRNGKey(2))

        obs_a, _ = env.reset(num_envs=4, env_params=params_a)
        obs_b, _ = env.reset(num_envs=4, env_params=params_b)

        self.assertFalse(jnp.array_equal(obs_a, obs_b))


if __name__ == "__main__":
    unittest.main()
