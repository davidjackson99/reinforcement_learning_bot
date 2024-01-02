import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS
from .parameters import TRANSITION_HISTORY_SIZE, RECORD_ENEMY_TRANSITIONS, LEARNING_RATE, DISCOUNT, game_rewards

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Events
INBOMBRADIUS_EVENT = "INBOMBRADIUS"
GOODPLACEMENT_EVENT = "GOODPLACEMENT"
BADPLACEMENT_EVENT = "BADPLACEMENT"
NOBOMB_EVENT = "NOBOMB"
MOREBOMBDIS_EVENT = "MOREBOMBDIS"
LESSBOMBDIS_EVENT = "LESSBOMBDIS"
ESCAPED_EVENT = "ESCAPED"
BACKIN_EVENT = "BACKIN"
BOMBPUSH_EVENT = "BOMBPUSH"


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

    # Idea: Add your own events to hand out rewards
    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        old_features = state_to_features(self, old_game_state)
        new_features = state_to_features(self, new_game_state)

        if np.sum(new_features[4])  != 0:
            events.append(INBOMBRADIUS_EVENT)
            if self_action == 'WAIT':
                events.append(BOMBPUSH_EVENT)


        if new_features[5] == 1 and self_action == 'WAIT':
            events.append(BOMBPUSH_EVENT)
        elif new_features[5] == 2 and self_action == 'UP':
            events.append(BOMBPUSH_EVENT)
        elif new_features[5] == 3 and self_action == 'DOWN':
            events.append(BOMBPUSH_EVENT)
        elif new_features[5] == 4 and self_action == 'RIGHT':
            events.append(BOMBPUSH_EVENT)
        elif new_features[5] == 5 and self_action == 'LEFT':
            events.append(BOMBPUSH_EVENT)


        field = np.transpose(new_game_state['field'])
        new_bomb = new_game_state['bombs']

        if new_bomb:
            bomb_surroundings = (field[new_bomb[0][0][1] - 1, new_bomb[0][0][0]],  # above
                                 field[new_bomb[0][0][1] + 1, new_bomb[0][0][0]],  # below
                                 field[new_bomb[0][0][1], new_bomb[0][0][0] + 1],  # right
                                 field[new_bomb[0][0][1], new_bomb[0][0][0] - 1])  # left


        if old_features[4] == 1 and new_features[4] == 2:
            events.append(MOREBOMBDIS_EVENT)
        elif old_features[4] == 2 and new_features[4] == 1:
            events.append(LESSBOMBDIS_EVENT)

        if old_features[4] == 2 and new_features[4] == 3:
            events.append(MOREBOMBDIS_EVENT)
        elif old_features[4] == 3 and new_features[4] == 2:
            events.append(LESSBOMBDIS_EVENT)

        if old_features[4] == 3 and new_features[4] == 4:
            events.append(ESCAPED_EVENT)
        elif old_features[4] == 4 and new_features[4] == 3:
            events.append(BACKIN_EVENT)

        if old_features[4] != 0 and new_features[4] == 0:
            events.append(ESCAPED_EVENT)

        if old_features[4] == 0 and new_features[4] != 0:
            events.append(BACKIN_EVENT)

        if old_features[4] == new_features[4]:
            events.append(INBOMBRADIUS_EVENT)


        if new_bomb and 1 in bomb_surroundings:
            events.append(GOODPLACEMENT_EVENT)
        else:
            events.append(BADPLACEMENT_EVENT)

        if new_bomb and self_action == 'BOMB':
            events.append(NOBOMB_EVENT)


        reward = reward_from_events(self, events)
        transition = Transition(old_features, self_action, new_features, reward)


        if e.CRATE_DESTROYED in events:
            self.q_table[old_features][ACTIONS.index(self_action)] = game_rewards[e.CRATE_DESTROYED]
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
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
