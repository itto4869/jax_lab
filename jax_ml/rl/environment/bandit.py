import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import Tuple

@dataclass
class BanditState:
    """
    A dataclass that represents the state of the bandit environment.
    
    Attributes:
        n_arms: number of arms
        expected_value: expected value of each arm
        step: number of current step
    """
    n_arms: jnp.int32
    expected_value: jnp.ndarray
    step: jnp.int32

class Bandit:
    def __init__(self):
        pass
    
    def init(self, key: jax.Array, n_arms: jnp.int16) -> BanditState:
        """
        Initialize the bandit environment.
        
        Args:
            key: PRNG key
            n_arms: number of arms
        
        Returns:
            state: initial state
        
        Example:
            >>> bandit = Bandit()
            >>> n_arms = 10 # number of arms.
            >>> batch_size = 2 # batch size
            >>> key = jax.random.PRNGKey(0)
            >>> keys = jax.random.split(key, batch_size)
            >>> state = jax.vmap(jax.jit(bandit.init, static_argnums=(1,)), in_axes=(0, None))(keys, n_arms))
            >>> print(state)
            BanditState(n_arms=Array([10, 10], dtype=int32, weak_type=True), expected_value=Array([[-2.6105583 ,  0.03385283,  1.0863333 , -1.4802988 ,  0.48895672,
                        1.062516  ,  0.54174834,  0.0170228 ,  0.2722685 ,  0.30522448],
                    [-0.38812608, -0.04487164, -2.0427258 ,  0.07932311,  0.33349916,
                        0.7959976 , -1.4411978 , -1.6929979 , -0.37369204, -1.5401139 ]],      dtype=float32), step=Array([0, 0], dtype=int32))
        """
        return _init(key, n_arms)

    def step(self, key: jax.Array, state: BanditState, action: jnp.int16) -> Tuple[BanditState, jnp.float32]:
        """
        Step the bandit environment.
        
        Args:
            key: PRNG key
            state: current state
            action: current action
        
        Returns:
            next_state: next state
            reward: Expected value state.expected_value[action] value sampled from a normal distribution with variance 1.
            
        Example:
            >>> bandit = Bandit()
            >>> n_arms = 10 # number of arms.
            >>> batch_size = 2 # batch size
            >>> key = jax.random.PRNGKey(0)
            >>> keys = jax.random.split(key, batch_size)
            >>> state = jax.vmap(jax.jit(bandit.init, static_argnums=(1,)), in_axes=(0, None))(keys, n_arms)
            >>> #action = get_action(state) # you define this function
            >>> action = jnp.array([0, 1]) # action example
            >>> next_state, reward = jax.vmap(jax.jit(bandit.step))(keys, state, action)
            >>> print(next_state, reward)
            BanditState(n_arms=Array([10, 10], dtype=int32, weak_type=True), expected_value=Array([[-2.6105583 ,  0.03385283,  1.0863333 , -1.4802988 ,  0.48895672,
                        1.062516  ,  0.54174834,  0.0170228 ,  0.2722685 ,  0.30522448],
                    [-0.38812608, -0.04487164, -2.0427258 ,  0.07932311,  0.33349916,
                        0.7959976 , -1.4411978 , -1.6929979 , -0.37369204, -1.5401139 ]],      dtype=float32), step=Array([1, 1], dtype=int32)) [-2.4666677 -1.2964104]
            """
        return _step(key, state, action)

def _init(key: jax.Array, n_arms: jnp.int16) -> BanditState:
    expected_value = jax.random.normal(key, (n_arms,))
    return BanditState(n_arms, expected_value, jnp.int32(0))

def _step(key: jax.Array, state: BanditState, action: jnp.int16) -> Tuple[BanditState, jnp.float32]:
    reward = jax.random.normal(key, shape=(), dtype=jnp.float32) + state.expected_value[action]
    next_state = BanditState(state.n_arms, state.expected_value, state.step + 1)
    return next_state, reward