import jax
import jax.numpy as jnp
from typing import Tuple, Union, List
from flax.struct import dataclass
from termcolor import colored
import numpy as np

@dataclass
class BoardState:
    board: jnp.ndarray
    current_player: jnp.int32
    valid_actions: jnp.ndarray
    done: jnp.bool_

class TicTacToe:
    def __init__(self):
        pass
    
    @staticmethod
    def init(key: jax.Array) -> BoardState:
        state = _init(key)
        return state
    
    @staticmethod
    def step(state: BoardState, action: jnp.int8) -> Union[Tuple[BoardState, jnp.float32], Tuple[BoardState, jnp.float32]]:
        state, reward = _step(state, action)
        return state, reward

    @staticmethod
    def reset(key: jax.Array) -> BoardState:
        state = _init(key)
        return state
    
    @staticmethod
    def render(states: BoardState) -> List[str]:
        visualed_boards = _render(states)
        return visualed_boards

def _init(key: jax.Array) -> BoardState:
    board = jnp.zeros((9,), dtype=jnp.int32)
    current_player = jnp.int32(jax.random.bernoulli(key))
    valid_actions = _get_valid_actions(board)
    return BoardState(board, current_player, valid_actions, jnp.bool_(False))

def _get_valid_actions(board: jnp.ndarray) -> jnp.ndarray:
    valid_actions = jnp.equal(board, 0)
    return valid_actions

def _step(state: BoardState, action: jnp.int32) -> Union[Tuple[BoardState, jnp.float32], Tuple[BoardState, jnp.ndarray]]:
    next_state = jax.lax.cond(
        state.done,
        lambda _: state,
        lambda _: _next_state(state, action),
        None,
    )
    reward = _reward(next_state.current_player, next_state.done)
    return next_state, reward

def _next_state(state: BoardState, action: jnp.int32) -> BoardState:
    next_board = state.board.at[action].set(state.current_player + 1)
    next_player = 1 - state.current_player
    valid_actions = _get_valid_actions(next_board)
    done = _is_done(next_board)
    next_state = BoardState(next_board, next_player, valid_actions, done)
    return next_state

def _is_done(board: jnp.ndarray) -> jnp.bool_:
    index = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [0, 3, 6], [1, 4, 7], [2, 5, 8],
                       [0, 4, 8], [2, 4, 6]])
    is_done_0 = _is_done_player(board, index, 0)
    is_done_1 = _is_done_player(board, index, 1)
    is_done = jnp.logical_or(is_done_0, is_done_1)
    return is_done

def _is_done_player(board: jnp.ndarray, index: jnp.ndarray, player_id: jnp.int32) -> jnp.bool_:
    return jnp.any(jnp.all(jnp.equal(board[index], player_id+1), axis=1))

def _reward(next_player: jnp.int32, done: jnp.bool_) -> Union[jnp.float32, jnp.ndarray]:
    # player 0 が勝った場合1、player 1 が勝った場合-1
    # stateは次の状態になっているので、state.current_playerも進んでいる
    reward = jax.lax.cond(
        next_player,
        lambda done: done * 1.0,
        lambda done: done * -1.0, 
        done)
    return reward

def _render(states: BoardState):
    batch_size = states.board.shape[0]
    visual_boards = []

    for i in range(batch_size):
        board = states.board[i].reshape((3, 3))
        visual_board = np.where(board == 1, 'X', np.where(board == 2, 'O', ' '))
        display_board = _format_board(visual_board)
        visual_boards.append(display_board)

    return visual_boards

def _format_board(board):
    formatted_board = ""
    for i, row in enumerate(board):
        formatted_board += " | ".join(row)
        if i < 2:
            formatted_board += "\n---------\n"
    return formatted_board