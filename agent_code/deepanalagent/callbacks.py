import os
import pickle
import random
from .classes import ReplayBuffer, build_dqn
from tensorflow.keras.models import load_model
import numpy as np



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our models is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.n_episodes = 15000
    self.epsilon_min = 0.02
    self.bomb = None
    if self.train:
        self.epsilon = 0.95
        self.epsilon_cur = 0.95
    else:
        self.epsilon_cur = 0.1
    self.action_space = ACTIONS
    self.model_file = "model.h5"

    self.input_dims = 675
    n_actions = len(ACTIONS)
    alpha = 0.0005


    if self.train:
        self.logger.info("Setting up models from scratch.")
        self.q_eval = build_dqn(alpha, n_actions, self.input_dims, 256, 256)
    else:
        self.logger.info("Loading models from saved state.")
        self.q_eval = load_model(self.model_file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    rand = np.random.random()
    if rand < self.epsilon_cur:
        action = np.random.choice(self.action_space, p=[.2, .2, .2, .2, .1, .1])
    else:
        state = state_to_features(self, game_state)
        state = state[np.newaxis, :]
        actions = self.q_eval.predict(state)
        action = self.action_space[np.argmax(actions)]

    return action


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your models, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state['field']
    bombs = game_state['bombs']

    if self.bomb is not None:
        for b in bombs:
            if b[0] == self.bomb[0]:
                self.bomb = b
                break

    # For example, you could construct several channels of equal shape, ...
    player_field = np.zeros(field.shape)
    object_field = game_state['field']
    danger_field = np.zeros(field.shape, dtype=np.float16)

    agent = game_state['self'][3]

    player_field[agent] = 1
    for other in game_state['others']:
        player_field[other[3]] = -1
    player_field = player_field[1:16, 1:16].reshape(225, )

    object_field = np.where(object_field == -1.0, -0.5,object_field)
    object_field = np.where(object_field == 0.0, 0.5, object_field)
    object_field = np.where(object_field == 1.0, 0.0, object_field)
    for coin in game_state['coins']:
        object_field[coin] = 1
    for bomb in bombs:
        object_field[bomb[0]] = -1
        if bomb[0] == agent:
            self.bomb = bomb

    object_field = object_field[1:16, 1:16].reshape(225, )

    for bomb in bombs:

        if self.bomb is not None and bomb == self.bomb:
            danger = (bomb[1] + 1) / 4
        else:
            danger = -(bomb[1] + 1) / 4
        pos = bomb[0]
        danger_field[pos] = danger
        # check above
        for i in range(1, 4):
            if field[pos[0], pos[1] - i] != -1:
                danger_field[pos[0] , pos[1] - i] = danger
            else:
                break
        # check below
        for i in range(1, 4):
            if field[pos[0], pos[1] + i] != -1:
                danger_field[pos[0], pos[1] + i] = danger
            else:
                break
        # check right
        for i in range(1, 4):
            if field[pos[0] + i, pos[1]] != -1:
                danger_field[pos[0] + i, pos[1]] = danger
            else:
                break
        # check left
        for i in range(1, 4):
            if field[pos[0] - i, pos[1]] != -1:
                danger_field[pos[0] - i, pos[1]] = danger
            else:
                break

    danger_field = danger_field[1:16, 1:16].reshape(225, )

    return np.concatenate((player_field, object_field, danger_field))
