import pickle
import numpy as np
from collections import namedtuple, deque
from typing import List
from .parameters import LEARNING_RATE, GAMMA, game_rewards, SAVE_EVERY, freq
import events as e
import settings as s
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.all_rewards = []
    self.round_reward = 0

    self.survived_rounds = []

    self.round_scores = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is not None:

        old_features = self.current_features
        new_features = state_to_features(self, new_game_state)
        symmetry_features = generalize_feature(old_features, new_features, self_action)
        self.current_features = new_features

        events = getArtificialRewards(self, old_game_state, new_game_state, old_features, new_features, self_action, events)
        reward = reward_from_events(self, events)
        self.round_reward += reward


        for sym in symmetry_features:
            old_feat = sym[0][0].reshape(1, -1)
            new_feat = sym[0][1].reshape(1, -1)
            action = sym[1]
            self.qfunction.remember(old_feat, ACTIONS.index(action), reward, new_feat, False)

        self.qfunction.experience_replay()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state['step'] < 400 and e.KILLED_SELF not in events:
        events.append('FINISHED_ROUND')

    reward = - reward_from_events(self, events)
    self.round_reward += reward

    symmetry_features = generalize_feature(self.current_features, self.current_features, last_action)

    for sym in symmetry_features:
        old_feat = sym[0][0].reshape(1, -1)
        new_feat = sym[0][1].reshape(1, -1)
        action = sym[1]
        self.qfunction.remember(old_feat, ACTIONS.index(action), reward, new_feat, True)

    self.qfunction.experience_replay()

    self.all_rewards.append(self.round_reward)
    self.round_reward = 0

    self.round_scores.append(last_game_state['self'][1])

    self.survived_rounds.append(last_game_state['step'])

    new_eps = self.epsilon_start * (1 - last_game_state['round'] / self.n_episodes)
    self.epsilon_cur = new_eps if new_eps > self.epsilon_min else self.epsilon_min

    # if last_game_state['self'][1] >= 8:
    #     with open("models/highest.pkl", "wb") as file:
    #         pickle.dump(self.q_table, file)
    #     for k in range(1000):
    #         print("YYYYY")
    # Store the models
    if last_game_state['round'] % SAVE_EVERY == 0:
        with open("models/my-saved-models.pkl", "wb") as file:
            pickle.dump(self.qfunction, file)
        with open("progress.pt", "wb") as file:
            pickle.dump([self.all_rewards, self.round_scores, self.survived_rounds], file)


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distanceDev(oldpos, newpos, objectpos):
    a = distance(oldpos, objectpos)
    b = distance(newpos, objectpos)
    return b - a

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


def getArtificialRewards(self, oldgame, newgame, oldfeatures, newfeatures, action, events):
    # old_surroundings = oldfeatures[0:4]
    old_target = oldfeatures[4]
    # old_bombflag = oldfeatures[5]
    old_bombsight = oldfeatures[6:]
    new_bombsight = newfeatures[6:]
    # if oldfeatures[0:4] == (1,1,1,1) and action != 'BOMB':
    #     events.append(e.INVALID_ACTION)

    if oldfeatures[5] == 1 and self.bomb is not None:
        d = distanceDev(oldgame['self'][3], newgame['self'][3],
                        self.bomb[0][::-1])  # remember that some coords are transposed VERFICKT

        if d > 0:
            events.append('MORE_DISTANCE_BOMB')
        else:
            # todo case wait
            events.append(e.LE_DISTANCE_BOMB)
        if old_bombsight.count(1) > new_bombsight.count(1):
            events.append(e.DODGED_BOMB)
        if old_bombsight.count(1) == 0 and action == 'WAIT':
            events.append(e.GOOD_WAIT)
    elif oldfeatures[5] == 0:
        if oldfeatures[4] != 0 and oldfeatures[oldfeatures[4] - 1] == 1 and action == 'BOMB':
            events.append(e.PERFECT_BOMB)
        elif oldfeatures[4] != 0 and oldfeatures[oldfeatures[4] - 1] == 1 and action != 'BOMB':
            events.append(e.SHOULD_BOMB)
        elif action == 'BOMB':
            events.append(e.BAD_BOMB)
        if oldfeatures[oldfeatures[4] - 1] != 1:
            if move_to_index[action] == old_target:
                events.append(e.RIGHT_DIRECTION)
            else:
                events.append(e.WRONG_DIRECTION)

    # elif old_bombsight.count(1) == new_bombsight.count(1) and new_bombsight.count(1) > 0:
    #     #     #     events.append('BOMB_STILL_IN_SIGHT')

    return events

dir_to_index = {
    (0, 0): 0,  # undefined
    (-1, 0): 1,  # top
    (0, 1): 2,  # right
    (1, 0): 3,  # bottom
    (0, -1): 4  # left
}

move_to_index = {
    'WAIT': 0,  # undefined
    'UP': 1,  # top
    'RIGHT': 2,  # right
    'DOWN': 3,  # bottom
    'LEFT': 4,  # left
    'BOMB': 5
}

target_to_move = {
    0: 'WAIT',
    1: (-1, 0),
    2: 'RIGHT',
    3: 'DOWN',
    4: 'LEFT'
}

def generalize_feature(old_features, new_features, action, justone=False):
    f = np.array([old_features, new_features])
    feature_list = [(f, action)]
    if justone:
        return feature_list
    surr = f[:, 0:4]
    target = f[:, 4:5]
    bombflag = f[:, 5:6]
    bombdir = f[:, 6:]
    # #first rotate
    for k in range(1, 4):
        target_ = np.where(target == 0, 0, ((target + k - 1) % 4) + 1)  # 1,2,3,4
        f_ = np.hstack((np.roll(surr, k, axis=1), target_, bombflag, np.roll(bombdir, k, axis=1)))
        action_ = action if action == 'WAIT' or action == 'BOMB' else ACTIONS[(ACTIONS.index(action) + k) % 4]
        feature_list.append((f_, action_))
    # mirror vertically
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(np.logical_or(target == 1, target == 3), (target + 2) % 4, target)
    f_[:, 0], f_[:, 2], f_[:, 6], f_[:, 8] = f[:, 2], f[:, 0], f[:, 8], f[:, 6]
    action_ = ACTIONS[(ACTIONS.index(action) + 2) % 4] if (action == 'UP' or action == 'DOWN') else action
    feature_list.append((f_, action_))

    # mirror horizontally
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(np.logical_or(target == 2, target == 4), (target % 4) + 2, target)
    f_[:, 1], f_[:, 3], f_[:, 7], f_[:, 9] = f[:, 3], f[:, 1], f[:, 9], f[:, 7]
    action_ = ACTIONS[(ACTIONS.index(action) + 2) % 4] if (action == 'RIGHT' or action == 'LEFT') else action
    feature_list.append((f_, action_))

    # mirror diagonally
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(target == 0, target, np.where(target % 2 == 1, target + 1, target - 1))
    f_[:, 1], f_[:, 0], f_[:, 3], f_[:, 2], f_[:, 7], f_[:, 6], f_[:, 9], f_[:, 8] = \
        f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 6], f[:, 7], f[:, 8], f[:, 9]
    help = {'UP': 'RIGHT', 'RIGHT': 'UP', 'DOWN': 'LEFT', 'LEFT': 'DOWN', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
    action_ = help[action]
    feature_list.append((f_, action_))

    f_ = np.copy(f)
    help = {1: 4, 2: 3, 3: 2, 4: 1, 0: 0}
    f_[:, 4:5] = np.array([[help[target[0, 0]]], [help[target[1, 0]]]])
    f_[:, 3], f_[:, 0], f_[:, 2], f_[:, 1], f_[:, 6], f_[:, 9], f_[:, 7], f_[:, 8] = \
        f[:, 0], f[:, 3], f[:, 1], f[:, 2], f[:, 9], f[:, 6], f[:, 8], f[:, 7]
    help = {'UP': 'LEFT', 'RIGHT': 'DOWN', 'DOWN': 'RIGHT', 'LEFT': 'UP', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
    action_ = help[action]
    feature_list.append((f_, action_))
    return feature_list