from __future__ import annotations

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from env_wrapper import BinPackGFN
from training_core import TrainState, make_train_step
from training_model import PolicyMLP


class IntegrationResetKeyRotationTest(unittest.TestCase):
    def test_env_reset_key_is_rotated_in_train_step(self) -> None:
        seed = 0
        rng = jax.random.PRNGKey(seed)

        env = BinPackGFN(
            max_num_items=6,
            max_num_ems=12,
            obs_num_ems=12,
            beta=10.0,
            dense_reward=False,
        )

        rng, env_init_key, net_init_key = jax.random.split(rng, 3)
        env_params = env.init(env_init_key)

        model = PolicyMLP(
            obs_dim=env.observation_space["shape"][0],
            num_actions=env.action_space.n,
            hidden_dim=8,
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
            num_envs=1,
        )

        train_step = make_train_step(residual_power=2, rng_split_count=3)

        state_1, _ = train_step(train_state)
        state_2, _ = train_step(state_1)

        self.assertFalse(jnp.array_equal(train_state.env_params.reset_key, state_1.env_params.reset_key))
        self.assertFalse(jnp.array_equal(state_1.env_params.reset_key, state_2.env_params.reset_key))


if __name__ == "__main__":
    unittest.main()
