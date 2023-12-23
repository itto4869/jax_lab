import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import Tuple

@dataclass
class BanditState:
    n_arms: jnp.int32
    expected_value: jnp.ndarray
    step: jnp.int32

def _init(key: jax.Array, n_arms: jnp.int16) -> BanditState:
    expected_value = jax.random.normal(key, (n_arms,))
    return BanditState(n_arms, expected_value, jnp.int32(0))

def _step(key: jax.Array, state: BanditState, action: jnp.int16) -> Tuple[BanditState, jnp.float32]:
    reward = jax.random.normal(key, shape=(), dtype=jnp.float32) + state.expected_value[action]
    next_state = BanditState(state.n_arms, state.expected_value, state.step + 1)
    return next_state, reward

init = jax.jit(jax.vmap(_init, in_axes=(0, None)), static_argnums=(1,))
step = jax.jit(jax.vmap(_step))

batch_size = 2
n_arms = 10

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, batch_size+1)
init_key, key = keys[:-1], keys[-1]
state = init(init_key, n_arms)
print(state)
keys = jax.random.split(key, batch_size+2)
step_key, action_key, key = keys[:-2], keys[-2], keys[-1]
action = jax.random.randint(action_key, (batch_size,), 0, n_arms)
state, reward = step(step_key, state, action)
print(state)
print(reward)