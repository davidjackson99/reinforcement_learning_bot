import pickle
import numpy as np
from collections import namedtuple, deque
from typing import List
from .classes import ReplayBuffer
from .parameters import SHOW_EVERY, game_rewards
import events as e
import settings as s
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    s.CRATE_DENSITY = 0.5
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.gamma = 0.95
    self.batch_size = 64
    mem_size = 100_000

    self.epsilon_history = []
    self.penalties = []
    self.round_penalties = 0
    self.round_scores =[]
    self.avg_penalties = []
    self.survived_steps = []

    self.last_features = None

    self.memory = ReplayBuffer(mem_size, self.input_dims, len(self.action_space))

def remember(self, state, action, reward, new_state, done):
    self.memory.store_transition(state, action, reward, new_state, done)

def learn(self):
    if self.memory.mem_cntr < self.batch_size:
        return
    state, action , reward, new_state, done = self.memory.sample_buffer(self.batch_size)
    action_values = np.arange(len(self.action_space), dtype=np.int8)
    action_indices = np.dot(action, action_values)

    q_eval = self.q_eval.predict(state)
    q_next = self.q_eval.predict(new_state)
    q_target = q_eval.copy()

    batch_index = np.arange(self.batch_size, dtype=np.int32)

    q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1)*done
    _ = self.q_eval.fit(state, q_target, verbose=0)


def save_model(self):
    self.q_eval.save(self.model_file)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self.last_features is None:
        self.last_features = state_to_features(self, new_game_state)

    else:

        new_features = state_to_features(self, new_game_state)

        reward = reward_from_events(self, events)
        self.round_penalties += reward

        remember(self, self.last_features, self.action_space.index(self_action), reward, new_features, False)

        self.last_features = new_features

        learn(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    last_features = state_to_features(self, last_game_state)

    reward = reward_from_events(self, events)

    self.round_penalties += reward

    remember(self, self.last_features, self.action_space.index(last_action), reward, last_features, True)

    self.last_features = None

    learn(self)

    self.epsilon_history.append(self.epsilon)
    self.penalties.append(self.round_penalties)

    i = last_game_state['round']
    avg_penalty = np.mean(self.penalties[max(0, i - 100):(i + 1)])
    self.avg_penalties.append(avg_penalty)
    print(f"episode: {i}, Survived Steps: {last_game_state['step']}, score: {last_game_state['self'][1]}, Reward: {self.round_penalties}, avg_Reward: {avg_penalty}, epsilon {self.epsilon_cur}")
    self.round_penalties = 0
    self.round_scores.append(last_game_state['self'][1])
    self.epsilon_cur = self.epsilon * (1 - i/self.n_episodes)
    self.survived_steps.append(last_game_state['step'])

    if i % SHOW_EVERY == 0 and i > 1:
        save_model(self)
        progress = [self.penalties, self.survived_steps, self.round_scores, self.avg_penalties]
        with open("progress.pt", "wb") as file:
            pickle.dump(progress, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
