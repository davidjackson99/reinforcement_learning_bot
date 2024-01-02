import os
import pickle
import random
import settings as s
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
useold = False

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
    self.target_coin = None
    if useold:
        #for now n of episodes and random prob has to be set by hand
        self.n_episodes = 1000
        #self.logger.info("Setting up models from scratch.")
        self.random_prob = 0.2
        self.random_decay = (0.005/self.random_prob)**(1/self.n_episodes)
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
    elif self.train or not os.path.isfile("my-saved-model.pt"):
        #self.logger.info("Setting up models from scratch.")
        self.n_episodes = 5000
        self.random_prob = 0.9
        self.random_decay = (0.005/self.random_prob)**(1/self.n_episodes)
        self.q_table = {}
        for i in range(-1, 1):
            for ii in range(-1, 1):
                for iii in range(-1, 1):
                    for iiii in range(-1, 1):
                        for iiiii in range(-14, 15):
                            for iiiiii in range(-14, 15):
                                self.q_table[(i, ii, iii, iiii, iiiii, iiiiii)] = [np.random.uniform(-1, 0) for i in range(5)]
    else:
        self.coins_ingame = 9
        self.random_prob = 0.1
        #self.logger.info("Loading models from saved state.")
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
    print(self.target_coin)
    if random.random() < self.random_prob:
        #print(self.random_prob)
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #not sure if this is allowed or helpful
        # p = getPossibleActions(game_state)
        action = np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1])
        self.logger.debug(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
        self.logger.debug("Choosing action purely at random.")
        # print(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
        return action
    action = ACTIONS[np.argmax(self.q_table[state])]
    # print(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
    self.logger.debug(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
    self.logger.debug("Querying models for action.")
    # print(self.q_table[state])
    # print(np.argmax(self.q_table[state]))

    #print("Round", game_state['round'], "Step", game_state['step'], "Action: ", action)
    return action


def getPossibleActions(game_state:dict):
    agent_pos = game_state['self'][3]
    field = np.transpose(game_state['field'])

    surroundings = (field[agent_pos[1] - 1, agent_pos[0]],  # above
                    field[agent_pos[1], agent_pos[0] + 1],  # right
                    field[agent_pos[1] + 1, agent_pos[0]],  # below
                    field[agent_pos[1], agent_pos[0] - 1])  # left

    c = 1
    actions = []
    for i in surroundings:
        if i == 0:
            actions.append(1)
            c += 1
        else:
            actions.append(0)
    actions.append(1)
    return [i/c for i in actions]




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

    surroundings = (field[agent_pos[1] - 1, agent_pos[0]],  # above
                    field[agent_pos[1] + 1, agent_pos[0]],  # below
                    field[agent_pos[1], agent_pos[0] + 1],  # right
                    field[agent_pos[1], agent_pos[0] - 1])  # left

    if self.target_coin is not None:
        #in play mode the target coin coin cant just be set to None again, so we have to count them
        if not self.train and len(game_state['coins']) == self.coins_ingame:
            return surroundings + (self.target_coin[0] - agent_pos[0], self.target_coin[1] - agent_pos[1])

        if self.train:
            return surroundings + (self.target_coin[0] - agent_pos[0], self.target_coin[1] - agent_pos[1])

    shortest_dis = 30
    if len(game_state['coins']) != 0:
        nearestcoin = None
        for coin in game_state['coins']:
            dis = distance(coin, agent_pos)
            if dis < shortest_dis:
                shortest_dis = dis
                nearestcoin = coin
        self.target_coin = nearestcoin
        return surroundings + (self.target_coin[0] - agent_pos[0], self.target_coin[1] - agent_pos[1])
    else:
        return surroundings + (0,0)



