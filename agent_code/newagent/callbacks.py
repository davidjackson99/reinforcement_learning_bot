import os
import pickle
import random
from random import shuffle
import numpy as np
from .parameters import EPSILON_MAX, EPSILON_MIN, MEMORY_SIZE, BATCH_SIZE, GAMMA, LEARNING_RATE
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from _collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
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

    self.n_episodes = 900
    self.epsilon_min = EPSILON_MIN
    self.bomb = None

    if self.train:
        self.epsilon_start = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_cur = EPSILON_MAX
    else:
        self.epsilon_cur = 0.05

    self.action_space = ACTIONS
    self.model_file = "models/models.pkl"
    self.current_features = None


    if self.train and not useold:
        print("Create new models")
        self.qfunction = QRegressor(len(ACTIONS))
    elif self.train and useold:
        print("Load old models")
        load_model(self.model_file)
    else:
        print("Load old models")
        self.qfunction = QRegressor(len(ACTIONS))
        self.qfunction.setweights(load_model(self.model_file))


def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def act(self, game_state: dict) -> str:
    if not self.train or (self.train and game_state['step'] == 1):
        features = state_to_features(self, game_state)
        if self.train:
            self.current_features = features
    else:
        features = self.current_features

    if random.random() < self.epsilon_cur:
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return action
    actions = self.qfunction.act(np.array(features).reshape(1, -1))
    action = self.action_space[np.argmax(actions[0])]
    return action

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


dir_to_index = {
    (0, 0) : 0, # undefined
    (-1, 0): 1, #top
    (0, 1):  2, #right
    (1, 0):  3, #bottom
    (0, -1): 4  #left
}


def state_to_features(self, game_state: dict) -> np.array:
    if game_state is None:
        return None

    # field objects
    agent_pos = game_state['self'][3][::-1]
    field = np.transpose(game_state['field'])
    coins = [c[::-1] for c in game_state['coins']]
    bombs = [(b[0][::-1], b[1]) for b in game_state['bombs']]
    # enemies = game_state['others']

    for bomb in bombs:
        field[bomb[0]] = -1

    if self.bomb is None:
        for bomb in bombs:
            if bomb[0] == agent_pos:
                self.bomb = [bomb[0], bomb[1]+1]
                break
    else:
        if self.bomb[1] == 0:
            self.bomb = None
        else:
            self.bomb = [self.bomb[0], self.bomb[1]-1]

    # vector of spaces around player, 0 free, 1 not free
    surroundings = np.array([field[agent_pos[0] - 1, agent_pos[1]],  # above
                             field[agent_pos[0], agent_pos[1] + 1],  # right
                             field[agent_pos[0] + 1, agent_pos[1]],  # below
                             field[agent_pos[0], agent_pos[1] - 1]])  # left
    surroundings = np.where(surroundings == -1, 1, surroundings)

    free_space = field == 0
    target_direction = 0

    for radius in [5, 10, 20]:

        nearcoins = [c for c in coins if distance(c, agent_pos) < radius]

        if len(nearcoins) != 0:
            coinstep = look_for_targets(free_space, agent_pos, nearcoins)
            target_direction = dir_to_index[(coinstep[0] - agent_pos[0], coinstep[1] - agent_pos[1])]
            break
        else:
            crate = findbestcratespot(field, agent_pos, radius)
            if crate is not None:
                target_direction = dir_to_index[(crate[0] -agent_pos[0], crate[1] - agent_pos[1])]
                break

    bombflag = 1 - int(game_state['self'][2])


    bombinsight = lookforbombs(agent_pos, field, bombs)

    return (tuple(surroundings) + (target_direction, bombflag) + bombinsight)


#todo image coords
def lookforbombs(agent, field, bombs):
    for b in bombs:
        field[b[0]] = 2
    top = 2 in field[max(0, agent[0] - 5):agent[0], agent[1]]
    right = 2 in field[agent[0], agent[1]+1:min(17, agent[1] + 6)]
    bottom = 2 in field[agent[0]+1:min(17, agent[0] + 6), agent[1]]
    left = 2 in field[agent[0], max(0, agent[1] - 5):agent[1]]
    # x2 = field[agent[0], max(0, agent[1] - 6):min(17, agent[1] + 7)]
    # y2 = field[max(0, agent[0] - 6):min(17, agent[0] + 7), agent[1]]
    return (int(top), int(right), int(bottom), int(left))


def findbestcratespot(field, agent, radius):
    mask_free, mask_crates = free_spaces(agent, radius, field)
    if not np.any(mask_crates):
        return None
    indices = np.array(np.where(mask_crates == True))
    return look_for_targets(mask_free, agent, list(map(tuple, indices.T)))


def free_spaces(index, radius, field):
    row = index[1]
    col = index[0]
    x = np.arange(0, 17)
    y = np.arange(0, 17)
    xx, yy = np.meshgrid(x, y)
    dist = (xx - row) ** 2 + (yy - col) ** 2
    mask_free = np.logical_and(dist < radius ** 2, np.logical_or(field == 0, field == 1))
    mask_crates = np.logical_and(dist < radius ** 2, field == 1)
    mask_free[index] = False

    return mask_free, mask_crates


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
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
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


class QRegressor:

    def __init__(self, action_space):
        self.exploration_rate = EPSILON_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        self.isFit = False

    def setweights(self, model):
        self.model = model
        self.isFit = True

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.isFit == True:
            q_values = self.model.predict(state)
        else:
            q_values = np.zeros(self.action_space).reshape(1, -1)
        return q_values

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, int(len(self.memory) / 1))
        X = []
        targets = []
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                if self.isFit:
                    q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                else:
                    q_update = reward
            if self.isFit:
                q_values = self.model.predict(state)
            else:
                q_values = np.zeros(self.action_space).reshape(1, -1)
            q_values[0][action] = q_update

            X.append(list(state[0]))
            targets.append(q_values[0])
        self.model.fit(X, targets)
        self.isFit = True
