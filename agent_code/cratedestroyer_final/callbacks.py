import os
import pickle
import random
from random import shuffle
import numpy as np
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
useoldmodel = False
modelfile = 'models/my-saved-models.pt'

def setup(self):
    self.ticker = 0
    #s.CRATE_DENSITY = 0.3
    #Definde global variables that are important for training and playing
    self.bomb = None
    self.previous_action = None
    self.previousflag = 0
    self.previousbomb = None
    np.random.seed()
    if useoldmodel == True:
        self.n_episodes = 4000
        self.epsilon_start = 0.02
        self.epsilon_cur = 0.02
        self.epsilon_min = 0.02
        self.current_features = None
        with open("models/highest-new.pt", "rb") as file:
            self.q_table = pickle.load(file)
        with open('models/ftable.pt', "rb") as file:
            self.f_table = pickle.load(file)
    elif self.train:
        self.n_episodes = 3000
        self.epsilon_start = 0.95
        self.epsilon_cur = 0.95
        self.epsilon_min = 0.05
        self.current_features = None
        self.q_table = np.random.uniform(-1, 0, (2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6))
        self.f_table = np.zeros((2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 6))
    else:
        self.epsilon_cur = 0
        self.logger.info("Loading models from saved state.")
        print('load models')
        with open("models/highest-new.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    #use saved features to save time
    if not self.train or (self.train and game_state['step'] == 1):
        features = state_to_features(self, game_state)
        if self.train:
            self.current_features = features
    else:
        features = self.current_features

    if random.random() < self.epsilon_cur:
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        if action == 'BOMB':
            self.previous_action = action
        else:
            self.previous_action = None
        return action
    # print(features)
    action = ACTIONS[np.argmax(self.q_table[features])]
    if action == 'BOMB':
        self.previous_action = action
    else:
        self.previous_action = None
    # print(action)
    # print("---------")
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

#TODO: Des auf 4 features
def getbombdirection(self, agent_pos, bombs): #When agent is out of bomb reach, get step towards bomb explosion
    oben, unten, links, rechts = 0,0,0,0
    for bomb in bombs:
        if distance(agent_pos, bomb[0]) < 5:
            if bomb[0]==agent_pos or bomb[0][0]==agent_pos[0] or bomb[0][1]==agent_pos[1]: #agent auf bombe, oder in gefahrenzone
                continue
            elif agent_pos[0]-1==bomb[0][0]: #ABOVE
                oben = 1
            elif agent_pos[1]+1==bomb[0][1]: #RIGHT
                rechts = 1
            elif agent_pos[0]+1==bomb[0][0]: #BELOW
                unten = 1
            elif agent_pos[1]-1==bomb[0][1]: #LEFT
                links = 1
    return (oben, rechts, unten, links)


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
                self.bomb = bomb
                break
    else:
        selfbomb = self.bomb
        self.bomb = None
        for bomb in bombs:
            if bomb[0] == selfbomb[0]:
                self.bomb = bomb
                break

    # vector of spaces around player, 0 free, 1 not free
    surroundings = np.array([field[agent_pos[0] - 1, agent_pos[1]],  # above
                             field[agent_pos[0], agent_pos[1] + 1],  # right
                             field[agent_pos[0] + 1, agent_pos[1]],  # below
                             field[agent_pos[0], agent_pos[1] - 1]])  # left
    surroundings = np.where(surroundings == -1, 1, surroundings)
    sackgassen = is_sackgasse(agent_pos, np.transpose(game_state['field']))


    if self.previous_action == 'BOMB':
        if sackgassen[0] == 1:
            surroundings[0] = 1
        if sackgassen[1] == 1:
            surroundings[1] = 1
        if sackgassen[2] == 1:
            surroundings[2] = 1
        if sackgassen[3] == 1:
            surroundings[3] = 1
        self.previous_action = None

    #print(surroundings)
    free_space = field == 0
    target_direction = 0
    #sackgassen = is_sackgasse(agent_pos, np.transpose(game_state['field']))


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

    if bombflag == 1:
        bombdirection = getbombdirection(self, agent_pos, bombs)
    else:
        bombdirection = (0,0,0,0)

    bombinsight = lookforbombs(agent_pos, field, bombs)

    return tuple(surroundings) + (target_direction, bombflag) + bombinsight + bombdirection # + sackgassen



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


def is_sackgasse(agent_pos, field): #extrem hässlig hardgecoded
    x = agent_pos[0]
    y = agent_pos[1]
    right = 0
    left = 0
    up = 0
    down = 0

    if field[x+1,y]==1 or field[x+1,y]==-1:
        down=0
    else:
        if field[x+1,y+1] == 0 or field[x+1,y-1] == 0:
            down = 0
        else:
            if field[x+2,y]==1 or field[x+2,y]==-1:
                down = 1
            else:
                if field[x+2,y+1] == 0 or field[x+2,y-1] == 0:
                    down = 0
                else:
                    if field[x +3, y] == 1 or field[x + 3, y] == -1:
                        down = 1
                    else:
                        if field[x + 3, y + 1] == 0 or field[x + 3, y - 1] == 0:
                            down = 0
                        else:
                            if field[x +4, y] == 1 or field[x + 4, y] == -1:
                                down = 1
                            else:
                                if field[x + 4, y + 1] == 0 or field[x + 4, y - 1] == 0:
                                    down = 0
                                else:
                                    if field[x + 5, y] == 1 or field[x + 5, y] == -1:
                                        down = 1


    if field[x-1,y]==1 or field[x-1,y]==-1:
        up=0
    else:
        if field[x-1,y+1] == 0 or field[x-1,y-1] == 0:
            up = 0
        else:
            if field[x-2,y]==1 or field[x-2,y]==-1:
                up = 1
            else:
                if field[x-2,y+1] == 0 or field[x-2,y-1] == 0:
                    up = 0
                else:
                    if field[x -3, y] == 1 or field[x - 3, y] == -1:
                        up = 1
                    else:
                        if field[x - 3, y + 1] == 0 or field[x - 3, y - 1] == 0:
                            up = 0
                        else:
                            if field[x -4, y] == 1 or field[x - 4, y] == -1:
                                up = 1
                            else:
                                if field[x - 4, y + 1] == 0 or field[x - 4, y - 1] == 0:
                                    up = 0
                                else:
                                    if field[x - 5, y] == 1 or field[x - 5, y] == -1:
                                        up = 1


    if field[x,y+1]==1 or field[x,y+1]==-1:
        right=0
    else:
        if field[x+1,y+1] == 0 or field[x-1,y+1] == 0:
            right = 0
        else:
            if field[x,y+2]==1 or field[x,y+2]==-1:
                right = 1
            else:
                if field[x+1,y+2] == 0 or field[x-1,y+2] == 0:
                    right = 0
                else:
                    if field[x, y+3] == 1 or field[x, y+3] == -1:
                        right = 1
                    else:
                        if field[x+1, y+3] == 0 or field[x-1, y+3] == 0:
                            right = 0
                        else:
                            if field[x, y+4] == 1 or field[x, y+4] == -1:
                                right = 1
                            else:
                                if field[x + 1, y + 4] == 0 or field[x - 1, y + 4] == 0:
                                    right = 0
                                else:
                                    if field[x, y + 5] == 1 or field[x, y + 5] == -1:
                                        right = 1


    if field[x,y-1]==1 or field[x,y-1]==-1:
        left=0
    else:
        if field[x+1,y-1] == 0 or field[x-1,y-1] == 0:
            left = 0
        else:
            if field[x,y-2]==1 or field[x,y-2]==-1:
                left = 1
            else:
                if field[x+1,y-2] == 0 or field[x-1,y-2] == 0:
                    left = 0
                else:
                    if field[x, y-3] == 1 or field[x, y-3] == -1:
                        left = 1
                    else:
                        if field[x+1, y-3] == 0 or field[x-1, y-3] == 0:
                            left = 0
                        else:
                            if field[x, y-4] == 1 or field[x, y-4] == -1:
                                left = 1
                            else:
                                if field[x + 1, y - 4] == 0 or field[x - 1, y - 4] == 0:
                                    left = 0
                                else:
                                    if field[x, y - 5] == 1 or field[x, y - 5] == -1:
                                        left = 1


    return (up, right, down, left)
