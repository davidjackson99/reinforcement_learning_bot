import os
import pickle
import random
import settings as s
import numpy as np
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
useold = False


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    #print(start, targets, free_space)

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        #FOLGENDE ZEILE BIN ICH MIR UNSICHER
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[y][x]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


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
    s.CRATE_DENSITY = 0.3
    self.target_coin = None
    self.target_crate = None
    self.target_bomb = None
    self.target_deadend = None
    self.prev_pos = None
    self.bomb_ticker = 0

    if useold:
        #for now n of episodes and random prob has to be set by hand
        self.n_episodes = 10000
        #self.logger.info("Setting up models from scratch.")
        self.random_prob = 0.9
        self.random_decay = (0.005/self.random_prob)**(1/self.n_episodes)
        self.current_features = None
        with open("models/my-saved-models-20000.pt", "rb") as file:
            self.q_table = pickle.load(file)
    elif self.train or not os.path.isfile("models/my-saved-models-20000.pt"):
        #self.logger.info("Setting up models from scratch.")
        self.n_episodes = 25000
        self.random_prob = 0.9
        self.random_decay = (0.005/self.random_prob)**(1/self.n_episodes)
        self.q_table = np.random.uniform(-1, 0, (5,5,5,5,3,3,3,3,3,3,6,6))
        self.current_features = None
    else:
        self.coins_ingame = 9
        self.random_prob = 0.1
        #self.logger.info("Loading models from saved state.")
        with open("models/my-saved-models-20000.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if not self.train or (self.train and game_state['step'] == 1):
        state = state_to_features(self, game_state)
        if self.train:
            self.current_features = state
    else:
        state = self.current_features

    if random.random() < self.random_prob:
        #print(self.random_prob)
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #not sure if this is allowed or helpful
        # p = getPossibleActions(game_state)
        action = np.random.choice(ACTIONS, p=[.21, .21, .21, .21, 0.1, 0.06])

        #instantiate/countdown bomb
        if self.bomb_ticker != 0:
            if self.bomb_ticker == 5:
                self.bomb_ticker = 0
            else:
                self.bomb_ticker = self.bomb_ticker + 1
        else:
            if action == 'BOMB':
                self.bomb_ticker = 1

        self.logger.debug(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
        self.logger.debug("Choosing action purely at random.")
        # print(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
        return action

    action = ACTIONS[np.argmax(self.q_table[state])]

    # instantiate/countdown bomb
    if self.bomb_ticker != 0:
        if self.bomb_ticker == 5:
            self.bomb_ticker = 0
        else:
            self.bomb_ticker = self.bomb_ticker + 1
    else:
        if action == 'BOMB':
            self.bomb_ticker = 1

    # print(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
    self.logger.debug(f"-----Round {game_state['round']} Step {game_state['step']} Action: {action} ------")
    self.logger.debug("Querying models for action.")
    # print(self.q_table[state])
    # print(np.argmax(self.q_table[state]))
    #print("Round", game_state['round'], "Step", game_state['step'], "Action: ", action)
    return action


def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_surroundings(agent_plus, bombs, coins, crates, free_space):
    if (agent_plus[0], agent_plus[1]) in crates:
        return 1
    if (agent_plus[0], agent_plus[1]) in bombs:
        return 2
    if (agent_plus[0], agent_plus[1]) in coins:
        return 3
    if free_space[agent_plus[0], agent_plus[1]] == True:
        return 0
    else:
        return 4


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
    field = game_state['field']


    '''surroundings = (field[agent_pos[0] - 1, agent_pos[1]]+1,  # above
                    field[agent_pos[0] + 1, agent_pos[1]]+1,  # below
                    field[agent_pos[0], agent_pos[1] + 1]+1,  # right
                    field[agent_pos[0], agent_pos[1] - 1]+1)  # left'''


    free_space = field == 0

    coins = game_state['coins']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (field[x, y] == 0)
                 and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (field[x, y] == 1)]
    surroundings = (get_surroundings([agent_pos[0] - 1, agent_pos[1]], bomb_xys, coins, crates, free_space),
                    get_surroundings([agent_pos[0] + 1, agent_pos[1]], bomb_xys, coins, crates, free_space),
                    get_surroundings([agent_pos[0], agent_pos[1] + 1], bomb_xys, coins, crates, free_space),
                    get_surroundings([agent_pos[0], agent_pos[1] - 1], bomb_xys, coins, crates, free_space))


    '''if len(dead_ends) != 0:
        dead_endstep = look_for_targets(free_space, agent_pos, dead_ends)
        dead_enddirection = (dead_endstep[0]-agent_pos[0]+1, dead_endstep[1]-agent_pos[1]+1)
        self.target_deadend = dead_enddirection
    else:
        self.target_deadend = None
        dead_enddirection = (1,1)'''

    '''if len(bomb_xys) != 0:
        bombstep = look_for_targets(free_space, agent_pos, bomb_xys)
        bombdirection = (bombstep[0]-agent_pos[0]+1, bombstep[1]-agent_pos[1]+1)
        self.target_bomb = bombdirection
    else:
        self.target_bomb = None
        bombdirection = (1,1)'''

    if len(bomb_xys) != 0:
        bomb_pos = bomb_xys[0]
        if agent_pos[0] == bomb_pos[0]:
            x = 0
        else:
            x = 1
        if agent_pos[1] == bomb_pos[1]:
            y = 0
        else:
            y = 1
        bombdirection = (x,y)
        bombstep = look_for_targets(free_space, agent_pos, bomb_xys)
        self.target_bomb = (bombstep[0] - agent_pos[0] + 1, bombstep[1] - agent_pos[1] + 1)
    else:
        bombdirection = (2,2)
        self.target_bomb = None

    if len(crates) != 0:
        coinstep = look_for_targets(free_space, agent_pos, crates)
        cratedirection = (coinstep[0]-agent_pos[0]+1, coinstep[1]-agent_pos[1]+1)
        self.target_crate = cratedirection
    else:
        self.target_crate = None
        cratedirection = (1, 1)

    if len(coins) != 0:
        coinstep = look_for_targets(free_space, agent_pos, coins)
        coindirection = (coinstep[0]-agent_pos[0]+1, coinstep[1]-agent_pos[1]+1)
        self.target_coin = coindirection


        return surroundings + coindirection + cratedirection + bombdirection + (self.bomb_ticker,)

    else:
        self.target_coin = None
        return surroundings + (1,1) + cratedirection + bombdirection  + (self.bomb_ticker,)
