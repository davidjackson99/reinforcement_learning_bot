import os
import pickle
import random
from random import shuffle
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
useoldmodel = False
modelfile = 'models/my-saved-model.pt'


# Definde global variables that are important for training and playing

def setup(self):
    self.bomb = None
    self.previous_action = None
    np.random.seed()
    if useoldmodel == True:
        self.n_episodes = 2000
        self.epsilon_start = 0.00
        self.epsilon_cur = 0.00
        self.epsilon_min = 0.00
        self.current_features = None
        with open("models/highest-new.pt", "rb") as file:
            self.q_table = pickle.load(file)
        # with open('models/ftable.pt', "rb") as file:
        #     self.f_table = pickle.load(file)
        self.last_q_table = None
    elif self.train:
        self.n_episodes = 6000
        self.epsilon_start = 0.95
        self.epsilon_cur = 0.95
        self.epsilon_min = 0.02
        self.current_features = None
        self.q_table = np.random.uniform(-1, 0, (2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6))
        # self.f_table = np.zeros((2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 6))
        self.last_q_table = None
    else:
        self.score= []
        self.epsilon_cur = 0
        self.logger.info("Loading model from saved state.")
        print('load model')
        with open("models/my-saved-models.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    #use saved features to save time
    if not self.train or (self.train and game_state['step'] == 1):
        features = state_to_features(self, game_state)
        if self.train:
            self.current_features = features
        #print(features)
        #print(features[0:4], features[4], features[5], features[6:10], features[10:14])

    else:
        features = self.current_features


        # print(features[0:4], features[4], features[5], features[6:10], features[10:14])

    if random.random() < self.epsilon_cur:
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        if action == 'BOMB':
            self.previous_action = action
        return action
    action = ACTIONS[np.argmax(self.q_table[features])]
    """print(action)
    print(self.q_table[features])
    print("--------")"""
    if action == 'BOMB':
        self.previous_action = action
    return action

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def free_space_distance(free_space, start, goal):
    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(goal, start)))

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all goal, track closest
        d = np.sum(np.abs(np.subtract(goal, current))).min()
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
    if goal in dist_so_far:
        return dist_so_far[goal]
    else:
        return 100


dir_to_index = {
    (0, 0) : 0, # undefined
    (-1, 0): 1, #top
    (0, 1):  2, #right
    (1, 0):  3, #bottom
    (0, -1): 4  #left
}


def bombdangermap(field, bombs):
    danger = 2
    fields = []
    for b in bombs:
        dangers = field.copy()
        pos = b[0]
        dangers[pos] = danger
        for i in range(1, 4):  # above
            if dangers[max(0, pos[0] - i), pos[1]] != -1:
                dangers[max(0, pos[0] - i), pos[1]] = danger
            else:
                break
        for i in range(1, 4):  # right
            if dangers[pos[0], min(pos[1] + i, 16)] != -1:
                dangers[pos[0], min(pos[1] + i, 16)] = danger
            else:
                break
        for i in range(1, 4):  # down
            if dangers[min(pos[0] + i, 16), pos[1]] != -1:
                dangers[min(pos[0] + i, 16), pos[1]] = danger
            else:
                break
        for i in range(1, 4):  # left
            if dangers[pos[0], max(0, pos[1] - i)] != -1:
                dangers[pos[0], max(0, pos[1] - i)] = danger
            else:
                break
        fields.append(dangers)
    return fields

def getbombdirection(agent_pos, bombs, field): #When agent is out of bomb reach, get step towards bomb explosion
    oben, unten, links, rechts = 0,0,0,0
    dangers = bombdangermap(field, bombs)
    for b, d in zip(bombs, dangers):
        if d[agent_pos] == 2:  # agent auf bombe, oder in gefahrenzone
            continue
        elif d[agent_pos[0] - 1, agent_pos[1]] == 2:  # ABOVE
            oben = 1
        elif d[agent_pos[0], agent_pos[1] + 1] == 2:  # RIGHT
            rechts = 1
        elif d[agent_pos[0] + 1, agent_pos[1]] == 2:  # BELOW
            unten = 1
        elif d[agent_pos[0], agent_pos[1] - 1] == 2:  # LEFT
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
    enemies_pos = [e[:][3][::-1] for e in game_state['others']]   # = [xy for (n, s, b, xy) in game_state['others']]

    for bomb in bombs:
        field[bomb[0]] = -1

    for enemy in enemies_pos:
        field[enemy] = 1

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

    sackgassenfield = game_state['field'].copy()
    for bomb in bombs:
        sackgassenfield[bomb[0]] = -1
    for enemy in enemies_pos:
        sackgassenfield[enemy] = -1
    sackgassen = is_sackgasse(agent_pos, field)


    temp = surroundings.copy()
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
        if np.count_nonzero(surroundings) == 4:
            surroundings = temp.copy()

    free_space = field == 0
    target_direction = 0
    now_bomb = 0
    #sackgassen = is_sackgasse(agent_pos, np.transpose(game_state['field']))


    for radius in [2, 5, 10, 20]:

        nearcoins = [c for c in coins if distance(c, agent_pos) < radius]
        # nearenemy = [e for e in enemies_pos if free_space_distance(free_space, agent_pos, e) < 6]

        # if len(nearenemy) != 0:
        #     enemystep = look_for_targets(free_space, agent_pos, nearenemy)
        #     target_direction = dir_to_index[(enemystep[0] - agent_pos[0], enemystep[1] - agent_pos[1])]
        #     break
        if len(nearcoins) != 0:
            coinstep = look_for_targets(free_space, agent_pos, nearcoins)
            if coinstep != agent_pos:
                target_direction = dir_to_index[(coinstep[0] - agent_pos[0], coinstep[1] - agent_pos[1])]
                break
        crate = findbestcratespot(field, agent_pos, radius)
        if crate is not None:
            target_direction = dir_to_index[(crate[0] - agent_pos[0], crate[1] - agent_pos[1])]
            break

    bombflag = 1 - int(game_state['self'][2])


    bombdirection = getbombdirection(agent_pos, bombs, field)


    if self.bomb is not None and self.bomb[0] == agent_pos:
        bombinsight = (1,1,1,1)
    else:
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
    """
    Searches all crates in radius and returns the position of the nearest
    """
    mask_free, mask_crates = free_spaces(agent, radius, field)
    if not np.any(mask_crates):
        return None
    indices = np.array(np.where(mask_crates == True))
    return look_for_targets(mask_free, agent, list(map(tuple, indices.T)))


def free_spaces(index, radius, field):
    """
    Returns:
        - mask free: all tiles in the field that are free or crates
        - mask crates: all tiles that are in radius and crates
    """

    row = index[1]
    col = index[0]
    x = np.arange(0, 17)
    y = np.arange(0, 17)
    xx, yy = np.meshgrid(x, y)
    dist = (xx - row) ** 2 + (yy - col) ** 2
    mask_free = np.logical_or(field == 0, field == 1)
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


def is_sackgasse(agent_pos, field):
    '''
    Input: agent position and field
    output: shows if above, right, below, and left towards the agent there is a deadend

    This function was written quite ugly, but it does what it should: iterate through the immediate surroundings,
    and see if there is there is no safe place from the bomb in each direction

    '''
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
