import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_ml.rl.environment.tic_tac_toe import TicTacToe
import jax
import jax.numpy as jnp

def test_game_loop():
    env = TicTacToe()
    batch_size = 2
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size)
    
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    
    state = init(keys)
    
    # init test
    assert jnp.equal(state.current_player, jnp.array([0, 1], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([False, False], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[True, True, True, True, True, True, True, True, True],
                                                     [True, True, True, True, True, True, True, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(state.board, jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)).all()
    
    actions = jnp.array([0, 0], dtype=jnp.int32)
    state, reward = step(state, actions)
    assert jnp.equal(state.current_player, jnp.array([1, 0], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([False, False], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[False, True, True, True, True, True, True, True, True],
                                                     [False, True, True, True, True, True, True, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(reward, jnp.array([0., 0.], dtype=jnp.float32)).all()
    assert jnp.equal(state.board, jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [2, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)).all()
    
    actions = jnp.array([1, 1], dtype=jnp.int32)
    state, reward = step(state, actions)
    assert jnp.equal(state.current_player, jnp.array([0, 1], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([False, False], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[False, False, True, True, True, True, True, True, True],
                                                     [False, False, True, True, True, True, True, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(reward, jnp.array([0., 0.], dtype=jnp.float32)).all()
    assert jnp.equal(state.board, jnp.array([[1, 2, 0, 0, 0, 0, 0, 0, 0],
                                             [2, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)).all()
    
    actions = jnp.array([3, 3], dtype=jnp.int32)
    state, reward = step(state, actions)
    assert jnp.equal(state.current_player, jnp.array([1, 0], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([False, False], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[False, False, True, False, True, True, True, True, True],
                                                     [False, False, True, False, True, True, True, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(reward, jnp.array([0., 0.], dtype=jnp.float32)).all()
    assert jnp.equal(state.board, jnp.array([[1, 2, 0, 1, 0, 0, 0, 0, 0],
                                             [2, 1, 0, 2, 0, 0, 0, 0, 0]], dtype=jnp.int32)).all()
    
    actions = jnp.array([2, 2], dtype=jnp.int32)
    state, reward = step(state, actions)
    assert jnp.equal(state.current_player, jnp.array([0, 1], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([False, False], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[False, False, False, False, True, True, True, True, True],
                                                     [False, False, False, False, True, True, True, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(reward, jnp.array([0., 0.], dtype=jnp.float32)).all()
    assert jnp.equal(state.board, jnp.array([[1, 2, 2, 1, 0, 0, 0, 0, 0],
                                             [2, 1, 1, 2, 0, 0, 0, 0, 0]], dtype=jnp.int32)).all()
    
    # done test
    actions = jnp.array([6, 6], dtype=jnp.int32)
    state, reward = step(state, actions)
    assert jnp.equal(state.current_player, jnp.array([1, 0], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([True, True], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[False, False, False, False, True, True, False, True, True],
                                                     [False, False, False, False, True, True, False, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(reward, jnp.array([1., -1.], dtype=jnp.float32)).all()
    assert jnp.equal(state.board, jnp.array([[1, 2, 2, 1, 0, 0, 1, 0, 0],
                                             [2, 1, 1, 2, 0, 0, 2, 0, 0]], dtype=jnp.int32)).all()
    
    actions = jnp.array([8, 8], dtype=jnp.int32)
    state, reward = step(state, actions)
    assert jnp.equal(state.current_player, jnp.array([1, 0], dtype=jnp.int32)).all()
    assert jnp.equal(state.done, jnp.array([True, True], dtype=jnp.bool_)).all()
    assert jnp.equal(state.valid_actions, jnp.array([[False, False, False, False, True, True, False, True, True],
                                                     [False, False, False, False, True, True, False, True, True]], dtype=jnp.bool_)).all()
    assert jnp.equal(reward, jnp.array([1., -1.], dtype=jnp.float32)).all()
    assert jnp.equal(state.board, jnp.array([[1, 2, 2, 1, 0, 0, 1, 0, 0],
                                             [2, 1, 1, 2, 0, 0, 2, 0, 0]], dtype=jnp.int32)).all()