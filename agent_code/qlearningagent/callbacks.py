import os
import pickle
import random
import settings as s
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
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
    s.CRATE_DENSITY = 0
    if useold:
        self.logger.info("Setting up models from scratch.")
        self.random_prob = 0.15
        self.random_decay = 0.9997
        with open("my-saved-models-highest-new.pt", "rb") as file:
            self.q_table = pickle.load(file)
    elif self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up models from scratch.")
        self.random_prob = 0.9
        self.random_decay = 0.9997
        self.q_table = np.random.uniform(low=-1, high=0, size=(3, 3, 3, 3, 5))
    else:
        self.target_coin = None
        self.random_prob = 0.1
        self.logger.info("Loading models from saved state.")
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

    state = state_to_features(self, game_state)
    if random.random() < self.random_prob:
        #print(self.random_prob)
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1])
    self.logger.debug("Querying models for action.")
    # print(self.q_table[state])
    # print(np.argmax(self.q_table[state]))
    action = ACTIONS[np.argmax(self.q_table[state])]


    #print("Action: ", action)
    return action


def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

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
    for c in game_state['coins']:
        field[c[1], c[0]] = 1



    surroundings = (field[agent_pos[1]-1, agent_pos[0]]+1, #above
                    field[agent_pos[1] + 1, agent_pos[0]]+1, #below
                    field[agent_pos[1], agent_pos[0]+1]+1, #right
                    field[agent_pos[1], agent_pos[0]-1]+1)



    return surroundings




