import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
useold = True


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
    if useold:
        # for now n of episodes and random prob has to be set by hand
        self.n_episodes = 12000
        # self.logger.info("Setting up models from scratch.")
        self.random_prob = 0.2
        self.random_decay = (0.005 / self.random_prob) ** (1 / self.n_episodes)
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
    elif self.train or not os.path.isfile("my-saved-model.pt"):
        # self.logger.info("Setting up models from scratch.")
        self.n_episodes = 12000
        self.random_prob = 0.9
        self.random_decay = (0.005 / self.random_prob) ** (1 / self.n_episodes)
        self.q_table = np.random.uniform(-1, 0, (3, 3, 3, 3, 5, 6, 6))    # (3, 3, 3, 3, 5, 6, 6)
    else:
        self.coins_ingame = 9
        self.random_prob = 0.1
        # self.logger.info("Loading models from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    state = state_to_features(self, game_state)

    if not self.train or (self.train and game_state['step'] == 1):
        state = state_to_features(self, game_state)
        if self.train:
            self.current_features = state
    else:
        state = self.current_features

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.16, .16, .16, .16, .16, .2])

    if not self.train and random.random() < random_prob:
        return np.random.choice(ACTIONS, p=[.17, .17, .17, .17, .17, .15])

    self.logger.debug("Querying models for action.")

    action = ACTIONS[np.argmax(self.q_table[state])]
    # print(action)
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

    agent_pos = game_state['self'][3]
    field = np.transpose(game_state['field'])
    bombs = game_state['bombs']
    bomb_radius_bool = 0
    where_bomb = 0
    surroundings_coor_0 = np.asarray((agent_pos[0] - 1, agent_pos[1],  # above
                                      agent_pos[0] + 1, agent_pos[1],  # below
                                      agent_pos[0], agent_pos[1] + 1,  # right
                                      agent_pos[0], agent_pos[1] - 1,  # left
                                      agent_pos[0], agent_pos[1]       # self
                                      )).reshape((5,2))
    surroundings_coor_1 = np.asarray((agent_pos[0] - 2, agent_pos[1],  # above
                                      agent_pos[0] + 2, agent_pos[1],  # below
                                      agent_pos[0], agent_pos[1] + 2,  # right
                                      agent_pos[0], agent_pos[1] - 2,  # left
                                      )).reshape((4, 2))
    surroundings_coor_2 = np.asarray((agent_pos[0] - 3, agent_pos[1],  # above
                                      agent_pos[0] + 3, agent_pos[1],  # below
                                      agent_pos[0], agent_pos[1] + 3,  # right
                                      agent_pos[0], agent_pos[1] - 3,  # left
                                      )).reshape((4, 2))
    surroundings_coor_3 = np.asarray((agent_pos[0] - 4, agent_pos[1],  # above
                                      agent_pos[0] + 4, agent_pos[1],  # below
                                      agent_pos[0], agent_pos[1] + 4,  # right
                                      agent_pos[0], agent_pos[1] - 4,  # left
                                      )).reshape((4, 2))

    if bombs:
        if bombs[0][0] in surroundings_coor_0:
            bomb_radius_bool = 1
        elif bombs[0][0] in surroundings_coor_1:
            bomb_radius_bool = 2
        elif bombs[0][0] in surroundings_coor_2:
            bomb_radius_bool = 3
        elif bombs[0][0] in surroundings_coor_3:
            bomb_radius_bool = 4


        if np.array_equal(agent_pos, bombs[0][0]):
            where_bomb = 1  # dont WAIT
        elif np.array_equal(agent_pos, np.subtract(bombs[0][0], (-1,0))):
            where_bomb = 2  # dont go UP
        elif np.array_equal(agent_pos, np.subtract(bombs[0][0], (1,0))):
            where_bomb = 3  # dont go DOWN
        elif np.array_equal(agent_pos, np.subtract(bombs[0][0], (0,1))):
            where_bomb = 4  # dont go RIGHT
        elif np.array_equal(agent_pos, np.subtract(bombs[0][0], (0,-1))):
            where_bomb = 5  # dont go LEFT



    features = (field[agent_pos[1] - 1, agent_pos[0]],  # above
                    field[agent_pos[1] + 1, agent_pos[0]],  # below
                    field[agent_pos[1], agent_pos[0] + 1],  # right
                    field[agent_pos[1], agent_pos[0] - 1],  # left
                    bomb_radius_bool,   # bomb radius
                    where_bomb)         # bomb push

    return features
