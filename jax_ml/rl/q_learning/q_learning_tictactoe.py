import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jax_ml.rl.environment.tic_tac_toe import TicTacToe, BoardState
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from functools import partial

@dataclass
class Param:
    Q: jnp.ndarray
    epsilon: jnp.float32
    alpha: jnp.float32
    gamma: jnp.float32
    
@dataclass
class Buffer:
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray

# Agentの状態を初期化する関数
@jax.jit
def _init_param(
    epsilon: jnp.float32 = 0.3,
    alpha: jnp.float32 = 0.1,
    gamma: jnp.float32 = 0.9
    ) -> Param:
    return Param(
        Q=jnp.zeros((3 ** 9, 9), dtype=jnp.float32),
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma
    )

# パラメータを更新する関数
def _update(
    Q: jnp.ndarray,
    action: jnp.int32, 
    obs: jnp.ndarray, 
    next_obs: jnp.ndarray, 
    gamma: jnp.float32, 
    alpha: jnp.float32, 
    reward: jnp.int32,
    ) -> jnp.ndarray:
    current_index = _obs_to_index(obs)
    next_index = _obs_to_index(next_obs)
    next_max_value = jnp.max(Q[next_index], axis=-1)
    next_q_value = Q[current_index, action] + alpha * (reward + gamma * next_max_value - Q[current_index, action])
    Q = Q.at[current_index, action].set(next_q_value)
    return Q

class Agent:
    def __init__(self):
        pass

    def init(self,
             epsilon: jnp.float32,
             alpha: jnp.float32,
             gamma: jnp.float32
             ) -> Param:
        param = _init_param(epsilon=epsilon, alpha=alpha, gamma=gamma)
        return param
    
    def update(self, param: Param, buffer: Buffer) -> Param:
        updated_Q = _update(param.Q, buffer.action, buffer.obs, buffer.next_obs, param.gamma, param.alpha, buffer.reward)
        param = Param(Q=updated_Q, epsilon=param.epsilon, alpha=param.alpha, gamma=param.gamma)
        return param
    
    def get_action(self, param: Param, state: BoardState, keys: jax.Array) -> jnp.ndarray:
        return _get_action(state.board, param.Q, param.epsilon, keys, state.valid_actions)
    
    def save_Q(self, param: Param, path: str = "Q.npy"):
        _save_Q(param.Q, path)
    
    def load_Q(self, path: str = "Q.npy") -> jnp.ndarray:
        Q = _load_Q(path)
        return Q

def _get_action(obs: jnp.ndarray, Q: jnp.ndarray, epsilon: jnp.float32, key: jax.Array, valid_action: jnp.ndarray) -> jnp.ndarray:
    index = _obs_to_index(obs)
    action = _epsilon_greedy(key, Q[index], epsilon, valid_action)
    return action

def _epsilon_greedy(key: jax.Array, Q_state: jnp.ndarray, epsilon: jnp.float32, valid_actions: jnp.ndarray) -> jnp.int32:
    num_valid_actions = jnp.sum(valid_actions)
    sample_action_prob = jnp.ones((9,), dtype=jnp.float32) / num_valid_actions
    sample_action_prob = jnp.where(valid_actions, sample_action_prob, 0)
    random_choice = jax.random.bernoulli(key, p=epsilon)
    random_action = jax.random.categorical(key, sample_action_prob)
    Q_state = jnp.where(valid_actions, Q_state, -1)
    best_action = jnp.argmax(Q_state)
    action = jnp.where(random_choice, random_action, best_action)
    #jax.debug.print("action {action}", action=action)
    return action

def _obs_to_index(obs):
    index = _base3_to_decimal(obs)
    return index

def _base3_to_decimal(arr: jnp.ndarray):
    powers = jnp.arange(arr.shape[-1])
    decimal = jnp.sum(arr * (3 ** powers), axis=-1)
    return decimal

def _save_Q(Q: jnp.ndarray, path: str = "Q.npy"):
    jnp.save(path, Q)
    
def _load_Q(path: str = "Q.npy"):
    Q = jnp.load(path)
    return Q

def check_done(state, key):
    state = jax.lax.cond(
        state.done,
        lambda _: reset(key),
        lambda _: state,
        None,
    )
    return state

def convert_buffer_data(data1, data2, current_player, update_player):
    data = jax.lax.cond(
        current_player == update_player,
        lambda _: data1,
        lambda _: data2,
        None,
    )
    return data

def convert_buffer_reward(reward1, reward2, current_player, update_player):
    reward = jax.lax.cond(
        current_player == update_player,
        lambda _: reward1,
        lambda _: reward2,
        None,
    )
    reward = jax.lax.cond(
        update_player == 0,
        lambda _: reward,
        lambda _: -reward,
        None,
    )
    return reward

def eval(episodes, param, state_first, keys, update_player):
    win = 0
    lose = 0
    finish_episodes = 0
    i = 0
    while finish_episodes < episodes:
        actions_first = get_action(param, state_first, keys[:, i, 0, :])
        state_second, rewards_first = step(state_first, actions_first)
        
        actions_second = get_action(param, state_second, keys[:, i, 1, :])
        next_state_first, rewards_second = step(state_second, actions_second)
            
        actions_third = get_action(param, next_state_first, keys[:, i, 2, :])
        next_state_second, rewards_third = step(next_state_first, actions_third)
            
        buffer_obs = jnp.where(state_first.current_player.reshape(-1, 1) == update_player, state_first.board, state_second.board)
        buffer_next_obs = jnp.where(state_first.current_player.reshape(-1, 1) == update_player, next_state_first.board, next_state_second.board)
        buffer_actions = convert_buffer_data(actions_first, actions_second, state_first.current_player, update_player)
        buffer_rewards = convert_buffer_reward(rewards_second, rewards_third, state_first.current_player, update_player)
            
        buffer = Buffer(obs=buffer_obs, next_obs=buffer_next_obs, action=buffer_actions, reward=buffer_rewards)
        param = update(param, buffer)
        finish_episodes += jnp.sum(next_state_second.done)
        state_first = check_done(next_state_second, keys[:, i, 3, :])
        win += jnp.sum(buffer_rewards == 1)
        lose += jnp.sum(buffer_rewards == -1)
        i += 1
    return win, lose

env = TicTacToe()
agent = Agent()
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
check_done = jax.jit(jax.vmap(check_done))
#update = jax.jit(jax.vmap(agent.update, in_axes=(None, 0), out_axes=None))
update = jax.jit(agent.update)
get_action = jax.jit(jax.vmap(agent.get_action, in_axes=(None, 0, 0)))
convert_buffer_data = jax.jit(jax.vmap(convert_buffer_data, in_axes=(0, 0, 0, None)))
convert_buffer_reward = jax.jit(jax.vmap(convert_buffer_reward, in_axes=(0, 0, 0, None)))
reset = env.reset

def train():
    
    batch_size = 1024
    steps = 10000
    update_player = 0
    
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size+1)
    
    state_first = init(keys[:-1])
    agent = Agent()
    param = agent.init(0.1, 0.1, 0.9)
    keys = jax.random.split(keys[-1], batch_size*steps*4+1)
    print("keys", keys.shape)
    eval_keys = jax.random.split(keys[-1], batch_size*100*4)
    eval_keys = eval_keys.reshape((batch_size, 100, 4, -1))
    keys = keys[:-1].reshape((batch_size, steps, 4, -1))
    print("param.Q", param.Q.shape)
    for i in range(steps):
        actions_first = get_action(param, state_first, keys[:, i, 0, :])
        state_second, rewards_first = step(state_first, actions_first)
        
        actions_second = get_action(param, state_second, keys[:, i, 1, :])
        next_state_first, rewards_second = step(state_second, actions_second)
        
        actions_third = get_action(param, next_state_first, keys[:, i, 2, :])
        next_state_second, rewards_third = step(next_state_first, actions_third)
        
        buffer_obs = jnp.where(state_first.current_player.reshape(-1, 1) == update_player, state_first.board, state_second.board)
        buffer_next_obs = jnp.where(state_first.current_player.reshape(-1, 1) == update_player, next_state_first.board, next_state_second.board)
        buffer_actions = convert_buffer_data(actions_first, actions_second, state_first.current_player, update_player)
        buffer_rewards = convert_buffer_reward(rewards_second, rewards_third, state_first.current_player, update_player)
        
        buffer = Buffer(obs=buffer_obs, next_obs=buffer_next_obs, action=buffer_actions, reward=buffer_rewards)
        #print("buffer", buffer)
        #print("update_player", update_player)
        param = update(param, buffer)
        state_first = check_done(next_state_second, keys[:, i, 3, :])
        
        if i % 100 == 0:
            win, lose = eval(100, param, state_first, eval_keys, update_player)
            win_rate = win / (win + lose)
            print("win", win)
            print("lose", lose)
            print("win rate", win_rate)

            update_player = jax.lax.cond(
                win_rate > 0.55,
                lambda _: 1 - update_player,
                lambda _: update_player,
                None,
            )
            print("update_player", update_player)
    print("param.Q", param.Q)
    agent.save_Q(param)

def play():
    batch_size = 1
    env = TicTacToe()
    agent = Agent()
    Q = agent.load_Q("Q.npy")
    param = agent.init(0.0, 0.0, 0.0)
    param = Param(Q=Q, epsilon=param.epsilon, alpha=param.alpha, gamma=param.gamma)
    key = jax.random.PRNGKey(0)
    state_key, key = jax.random.split(key, batch_size+1)
    state = env.init(state_key)
    while True:
        key, get_action_key = jax.random.split(key)
        action = agent.get_action(param, state, get_action_key)
        state, reward = env.step(state, action)
        print("state", state.board)
        print("reward", reward)
        if state.done:
            break