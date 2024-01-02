import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .parameters import LEARNING_RATE, DISCOUNT, game_rewards, SAVE_EVERY, freq, delayed_events

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'symmetry'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 6  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.all_rewards = []
    self.round_reward = 0
    self.survived_rounds = []
    self.round_scores = []
    self.epsilons = []
    self.previousaction2 = None


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distanceDev(oldpos, newpos, objectpos):
    a = distance(oldpos, objectpos)
    b = distance(newpos, objectpos)
    return b - a


def getArtificialRewards(self, oldgame, newgame, oldfeatures, newfeatures, action, events, previous_action):
    # old_surroundings = oldfeatures[0:4]
    old_target = oldfeatures[4]
    # old_bombflag = oldfeatures[5]
    old_bombsight = oldfeatures[6:10]
    new_bombsight = newfeatures[6:10]
    if action == 'BOMB':
        if newfeatures[:4] == (1,1,1,1):
            events.append(e.BAD_BOMB)

    elif previous_action == 'BOMB':
        if oldfeatures[0] == 1 and action == 'UP':
            events.append(e.INVALID_ACTION)
        if oldfeatures[1] == 1 and action == 'RIGHT':
            events.append(e.INVALID_ACTION)
        if oldfeatures[2] == 1 and action == 'DOWN':
            events.append(e.INVALID_ACTION)
        if oldfeatures[3] == 1 and action == 'LEFT':
            events.append(e.INVALID_ACTION)

    if oldfeatures[10:14].count(1) != 0:
        if oldfeatures[10] == 1 and action == 'UP':
            events.append(e.INVALID_ACTION)
        elif oldfeatures[11] == 1 and action == 'RIGHT':
            events.append(e.INVALID_ACTION)
        elif oldfeatures[12] == 1 and action == 'DOWN':
            events.append(e.INVALID_ACTION)
        elif oldfeatures[13] == 1 and action == 'LEFT':
            events.append(e.INVALID_ACTION)

    if oldfeatures[5] == 1 and self.bomb is not None:
        d = distanceDev(oldgame['self'][3], newgame['self'][3],
                        self.bomb[0][::-1])  # remember that some coords are transposed VERFICKT

        # if d > 0:
        #     events.append('MORE_DISTANCE_BOMB')
        # else:
        #     # todo case wait
        #     events.append(e.LE_DISTANCE_BOMB)
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
    # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None:
        # Idea: Add your own events to hand out rewards
        # state_to_features is defined in callbacks.py

        '''if 'BOMB_EXPLODED' in events:
            print("BBOOOOMM")'''

        old_features = self.current_features
        new_features = state_to_features(self, new_game_state)
        symmetry_features = generalize_feature(old_features, new_features, self_action)
        self.current_features = new_features

        '''
        if self.bomb != None:
            if old_features[10] == 1 and self.previous_action == 'BOMB':
                if self_action == 'UP':
                    events.append(e.LE_DISTANCE_BOMB)
            if old_features[11] == 1 and self.previous_action == 'BOMB':
                if self_action == 'RIGHT':
                    events.append(e.LE_DISTANCE_BOMB)
            if old_features[10] == 1 and self.previous_action == 'BOMB':
                if self_action == 'DOWN':
                    events.append(e.LE_DISTANCE_BOMB)
            if old_features[10] == 1 and self.previous_action == 'BOMB':
                if self_action == 'LEFT':
                    events.append(e.LE_DISTANCE_BOMB)'''
        old_events = []
        eventscopy = events.copy()
        for e in eventscopy:
            if e in delayed_events:
                events.remove(e)
                old_events.append(e)

        # print("events", events)
        # print("old", old_events)

        events = getArtificialRewards(self, old_game_state, new_game_state, old_features, new_features, self_action,
                                      events, self.previousaction2)
        reward = reward_from_events(self, events)

        print(f"--------{old_game_state['step']}--{new_game_state['step']}-----------")
        print(self.bomb)
        print(f"Pos {old_game_state['self'][3][::-1]}---{new_game_state['self'][3][::-1]}")
        print(old_features)
        print(self.q_table[old_features])
        print(self_action)
        print(events)
        print(reward)
        print(new_features)
        print(self.round_reward)
        print("--------")
        transition = Transition(old_features, self_action, new_features, reward, symmetry_features)

        # if e.WAITED in events:
        #     print("")

        for sym in symmetry_features:
            old_feat = tuple(sym[0][0])
            new_feat = tuple(sym[0][1])
            action = sym[1]
            max_future_q = np.max(self.q_table[new_feat])
            current_q = self.q_table[old_feat][ACTIONS.index(action)]
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
            print(f"Currentq {current_q}, maxfutureq {max_future_q}, newq {new_q}")
            print(old_feat)
            self.q_table[old_feat][ACTIONS.index(action)] = new_q
            print(self.q_table[old_feat])

        if len(old_events) != 0:
            delayed_reward = reward_from_events(self, old_events)
            symmetry = self.transitions[0][4]
            print("-----------DELAYED_LEARNING")
            qfunction(self, symmetry, delayed_reward)

        self.transitions.append(transition)

        print("------")
        self.round_reward += reward
        self.previousaction2 = action


def qfunction(self, symmetry_features, reward):
    for sym in symmetry_features:
        old_feat = tuple(sym[0][0])
        new_feat = tuple(sym[0][1])
        action = sym[1]
        max_future_q = np.max(self.q_table[new_feat])
        current_q = self.q_table[old_feat][ACTIONS.index(action)]
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
        # print(f"Currentq {current_q}, maxfutureq {max_future_q}, newq {new_q}")
        # print(old_feat)
        self.q_table[old_feat][ACTIONS.index(action)] = new_q

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
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # if last_game_state['step'] < 400 and e.KILLED_SELF not in events:
    #     events.append('FINISHED_ROUND')

    reward = reward_from_events(self, events)
    self.round_reward += reward

    symmetry_features = generalize_feature(self.current_features, self.current_features, last_action)

    for sym in symmetry_features:
        old_feat = tuple(sym[0][0])
        action = sym[1]
        self.q_table[old_feat][ACTIONS.index(action)] = reward
        #self.f_table[old_feat][ACTIONS.index(action)] += 1

    self.all_rewards.append(self.round_reward)
    self.round_reward = 0
    self.round_scores.append(last_game_state['self'][1])
    self.survived_rounds.append(last_game_state['step'])

    new_eps = self.epsilon_start * (1 - last_game_state['round'] / self.n_episodes)
    self.epsilons.append(new_eps)
    self.epsilon_cur = new_eps if new_eps > self.epsilon_min else self.epsilon_min
    if last_game_state['self'][1]>= 8:
        with open("models/highest-new.pt", "wb") as file:
            pickle.dump(self.q_table, file)
        for k in range(1000):
            print("YYYYY")
    # Store the models
    if last_game_state['round'] % SAVE_EVERY == 0 or last_game_state['round'] == self.n_episodes:
        with open("models/my-saved-models.pt", "wb") as file:
            pickle.dump(self.q_table, file)
        with open("progress.pt", "wb") as file:
            pickle.dump([self.all_rewards, self.round_scores, self.survived_rounds, self.epsilons], file)
        with open("models/ftable.pt", "wb") as file:
            pickle.dump(self.f_table, file)
        print(freq)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            freq[event] += 1
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def generalize_feature(old_features, new_features, action):
    f = np.array([old_features, new_features])
    feature_list = [(f, action)]
    surr = f[:, 0:4]
    target = f[:, 4:5]
    bombflag = f[:, 5:6]
    bombdir = f[:, 6:10]
    bombgefahr = f[:,10:14]
    # #first rotate
    for k in range(1, 4):
        target_ = np.where(target == 0, 0, ((target + k - 1) % 4) + 1)  # 1,2,3,4
        f_ = np.hstack((np.roll(surr, k, axis=1), target_, bombflag, np.roll(bombdir, k, axis=1), np.roll(bombgefahr, k, axis=1)))
        action_ = action if action == 'WAIT' or action == 'BOMB' else ACTIONS[(ACTIONS.index(action) + k) % 4]
        feature_list.append((f_, action_))
    # mirror vertically
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(np.logical_or(target == 1, target == 3), (target + 2) % 4, target)
    f_[:, 0], f_[:, 2], f_[:, 6], f_[:, 8], f_[:, 10], f_[:, 12] = f[:, 2], f[:, 0], f[:, 8], f[:, 6], f[:, 12], f[:, 10]
    action_ = ACTIONS[(ACTIONS.index(action) + 2) % 4] if (action == 'UP' or action == 'DOWN') else action
    feature_list.append((f_, action_))

    # mirror horizontally
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(np.logical_or(target == 2, target == 4), (target % 4) + 2, target)
    f_[:, 1], f_[:, 3], f_[:, 7], f_[:, 9], f_[:, 11], f_[:, 13] = f[:, 3], f[:, 1], f[:, 9], f[:, 7], f[:, 13], f[:, 11]
    action_ = ACTIONS[(ACTIONS.index(action) + 2) % 4] if (action == 'RIGHT' or action == 'LEFT') else action
    feature_list.append((f_, action_))

    # mirror diagonally
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(target == 0, target, np.where(target % 2 == 1, target + 1, target - 1))
    f_[:, 1], f_[:, 0], f_[:, 3], f_[:, 2], f_[:, 7], f_[:, 6], f_[:, 9], f_[:, 8], f_[:, 11], f_[:, 10], f_[:, 13], f_[:, 12]= \
        f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 6], f[:, 7], f[:, 8], f[:, 9], f[:, 10], f[:, 11], f[:, 12], f[:, 13]
    help = {'UP': 'RIGHT', 'RIGHT': 'UP', 'DOWN': 'LEFT', 'LEFT': 'DOWN', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
    action_ = help[action]
    feature_list.append((f_, action_))

    f_ = np.copy(f)
    help = {1: 4, 2: 3, 3: 2, 4: 1, 0: 0}
    f_[:, 4:5] = np.array([[help[target[0, 0]]], [help[target[1, 0]]]])
    f_[:, 10:11] = np.array([[help[bombgefahr[0, 0]]], [help[bombgefahr[1, 0]]]])
    f_[:, 3], f_[:, 0], f_[:, 2], f_[:, 1], f_[:, 6], f_[:, 9], f_[:, 7], f_[:, 8], f_[:, 13], f_[:, 10], f_[:, 12], f_[:, 11]= \
        f[:, 0], f[:, 3], f[:, 1], f[:, 2], f[:, 9], f[:, 6], f[:, 8], f[:, 7], f[:, 10], f[:, 13], f[:, 11], f[:, 12]
    help = {'UP': 'LEFT', 'RIGHT': 'DOWN', 'DOWN': 'RIGHT', 'LEFT': 'UP', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
    action_ = help[action]
    feature_list.append((f_, action_))
    return feature_list
'''


def generalize_feature(old_features, new_features, action):
    f = np.array([old_features, new_features])
    feature_list = [(f, action)]
    surr = f[:, 0:4]
    target = f[:, 4:5]
    bombflag = f[:, 5:6]
    bombdir = f[:, 6:10]
    sackgassen = f[:, 10:]
    # #first rotate
    for k in range(1, 4):
        target_ = np.where(target == 0, 0, ((target + k - 1) % 4) + 1)  # 1,2,3,4
        f_ = np.hstack((np.roll(surr, k, axis=1), target_, bombflag, np.roll(bombdir, k, axis=1), np.roll(sackgassen, k, axis=1)))
        action_ = action if action == 'WAIT' or action == 'BOMB' else ACTIONS[(ACTIONS.index(action) + k) % 4]
        feature_list.append((f_, action_))
    # mirror vertically
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(np.logical_or(target == 1, target == 3), (target + 2) % 4, target)
    f_[:, 0], f_[:, 2], f_[:, 6], f_[:, 8], f_[:, 10], f_[:, 12] = f[:, 2], f[:, 0], f[:, 8], f[:, 6], f[:, 12], f[:, 10]
    action_ = ACTIONS[(ACTIONS.index(action) + 2) % 4] if (action == 'UP' or action == 'DOWN') else action
    feature_list.append((f_, action_))

    # mirror horizontally
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(np.logical_or(target == 2, target == 4), (target % 4) + 2, target)
    f_[:, 1], f_[:, 3], f_[:, 7], f_[:, 9], f_[:, 11], f_[:, 13]= f[:, 3], f[:, 1], f[:, 9], f[:, 7], f[:, 13], f[:, 11]
    action_ = ACTIONS[(ACTIONS.index(action) + 2) % 4] if (action == 'RIGHT' or action == 'LEFT') else action
    feature_list.append((f_, action_))

    # mirror diagonally
    f_ = np.copy(f)
    f_[:, 4:5] = np.where(target == 0, target, np.where(target % 2 == 1, target + 1, target - 1))
    f_[:, 1], f_[:, 0], f_[:, 3], f_[:, 2], f_[:, 7], f_[:, 6], f_[:, 9], f_[:, 8], f_[:, 11], f_[:, 10], f_[:, 13], f_[:, 12] = \
        f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 6], f[:, 7], f[:, 8], f[:, 9], f[:, 10], f[:, 11], f[:, 12], f[:, 13]
    help = {'UP': 'RIGHT', 'RIGHT': 'UP', 'DOWN': 'LEFT', 'LEFT': 'DOWN', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
    action_ = help[action]
    feature_list.append((f_, action_))

    f_ = np.copy(f)
    help = {1: 4, 2: 3, 3: 2, 4: 1, 0: 0}
    f_[:, 4:5] = np.array([[help[target[0, 0]]], [help[target[1, 0]]]])
    f_[:, 3], f_[:, 0], f_[:, 2], f_[:, 1], f_[:, 6], f_[:, 9], f_[:, 7], f_[:, 8], f_[:, 13], f_[:, 10], f_[:, 12], f_[:, 11] = \
        f[:, 0], f[:, 3], f[:, 1], f[:, 2], f[:, 9], f[:, 6], f[:, 8], f[:, 7], f[:, 10], f[:, 13], f[:, 11], f[:, 12]
    help = {'UP': 'LEFT', 'RIGHT': 'DOWN', 'DOWN': 'RIGHT', 'LEFT': 'UP', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
    action_ = help[action]
    feature_list.append((f_, action_))
    return feature_list
'''
