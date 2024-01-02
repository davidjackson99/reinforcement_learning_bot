import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS
from .parameters import TRANSITION_HISTORY_SIZE, RECORD_ENEMY_TRANSITIONS, LEARNING_RATE, DISCOUNT, game_rewards
import collections

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
LESSDISTANCE_EVENT = 'LESSDISTANCE'
SAMEDISTANCE_EVENT = 'SAMEDISTANCE'
MOREDISTANCE_EVENT = 'MOREDISTANCE'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.all_rewards = []
    self.round_reward = 0
    self.highest_reward = -10000000
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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

    if old_game_state is not None:
        # Idea: Add your own events to hand out rewards
        # state_to_features is defined in callbacks.py

        old_features = state_to_features(self, old_game_state)
        if e.COIN_COLLECTED in events:
            self.target_coin = None
        new_features = state_to_features(self, new_game_state)

        dist_old = np.sqrt(old_features[4]**2 + old_features[5]**2)
        dist_new = np.sqrt(new_features[4] ** 2 + new_features[5] ** 2)

        if dist_old < dist_new:
            events.append(MOREDISTANCE_EVENT)
        elif dist_old == dist_new:
            events.append(SAMEDISTANCE_EVENT)
        else:
            events.append(LESSDISTANCE_EVENT)


        """
        if len(self.transitions) > 2:
            repetitionMultiplier = recognizeRepetition(self, self_action)
            for k in range(repetitionMultiplier):
                events.append(REPETITION_EVENT)
        """

        reward = reward_from_events(self, events)
        # for k in events:
        #     print(k)
        # print(e.WAITED)
        # print("--------")
        transition = Transition(old_features, self_action, new_features, reward)

        # if e.WAITED in events:
        #     print("")

        if e.COIN_COLLECTED in events:                                                                # ???
            self.q_table[old_features][ACTIONS.index(self_action)] = game_rewards[e.COIN_COLLECTED]
        else:

            max_future_q = np.max(self.q_table[new_features])
            current_q = self.q_table[old_features][ACTIONS.index(self_action)]

            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
            self.q_table[old_features][ACTIONS.index(self_action)] = new_q
        self.round_reward += reward

        self.transitions.append(transition)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    print(self.round_reward, "coins: ", last_game_state['self'][1])
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))
    self.random_prob = self.random_prob * self.random_decay
    round = last_game_state['round']
    coins = last_game_state['self'][1]
    self.all_rewards.append(self.round_reward)
    self.logger.debug(f'Round {round}  --> Reward: {self.round_reward}, Coins: {coins}')
    if self.round_reward > self.highest_reward:
        self.highest_reward = self.round_reward
        with open("my-saved-model-highest.pt", "wb") as file:
            pickle.dump(self.q_table, file)
    self.round_reward = 0
    # Store the models

    if round == self.n_episodes:
        with open("all-rewards.pt", "wb") as file:
            pickle.dump(self.all_rewards, file)
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.q_table, file)


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
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def recognizeRepetition(self, action):
    opposite = {'UP': 'DOWN',
                'DOWN': 'UP',
                'RIGHT':'LEFT',
                'LEFT': 'RIGHT',
                'WAIT': ''}
    c = 0
    for t in range(len(self.transitions)-1,-1, -1):
        if self.transitions[t][1] == opposite[action]:
            c+=1
            action = self.transitions[t][1]
        else:
            return c

    return c