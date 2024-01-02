import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .parameters import LEARNING_RATE, DISCOUNT, game_rewards, SAVE_EVERY, delayed_events

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'symmetry'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 6  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.transitions = []
    self.all_rewards = []
    self.round_reward = 0
    self.survived_rounds = []
    self.round_scores = []
    self.avg_scores = []
    self.epsilons = []
    self.highest_score = 0
    self.convergence = []
    self.old_q_table = None
    self.repetion = self.n_episodes / 7 + 1
    self.repetion = self.n_episodes / 3 + 1


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

dir_to_index = {
    (0, 0): 0,  # undefined
    (-1, 0): 1,  # top
    (0, 1): 2,  # right
    (1, 0): 3,  # bottom
    (0, -1): 4  # left
}

move_to_index = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5,
}

target_to_move = {
    0: 'WAIT',
    1: (-1, 0),
    2: 'RIGHT',
    3: 'DOWN',
    4: 'LEFT'
}

def or_tuples(a, b):
    return tuple(np.logical_or(a, b).astype(int))

def and_tuples(a, b):
    return tuple(np.logical_or(a, b).astype(int))


def getArtificialRewards(oldfeatures, newfeatures, action, events):
    oldsurroundings = oldfeatures[0:4]
    newsurroundings = newfeatures[0:4]

    oldtarget = oldfeatures[4]
    newtarget = newfeatures[4]

    oldbombflag = oldfeatures[5]
    newbombflag = newfeatures[5]

    oldbombsight = oldfeatures[6:10]
    newbombsight = newfeatures[6:10]

    oldbombdanger = oldfeatures[10:14]
    newbombdanger = newfeatures[10:14]

    # oldenemysight = oldfeatures[14]
    # newenemysight = newfeatures[14]


    if oldbombsight.count(1) > 0:
        if np.count_nonzero(oldsurroundings) == 4:
            if action == 'WAIT':
                events.append(e.PERFECT_MOVE)
            else:
                events.append(e.BAD_MOVE)
        elif oldbombdanger.count(0) == 0:  # follow target if no safespots available
            if move_to_index[action] == oldtarget - 1:
                events.append(e.PERFECT_MOVE)
            else:
                events.append(e.BAD_MOVE)
        elif action != 'WAIT' and action != 'BOMB':
            space = or_tuples(oldbombsight, or_tuples(oldbombdanger, oldsurroundings))
            if space[move_to_index[action]] == 0:  # walked in direction of no danger or barrier
                if space[oldtarget -1] == 0 and move_to_index[action] == oldtarget - 1: #target was free and we moved towards it
                    events.append(e.PERFECT_MOVE)
                elif space[oldtarget -1] != 0 and move_to_index[action] != oldtarget - 1: #target was not free and we didnt go there
                    events.append(e.PERFECT_MOVE)
                else:
                    events.append(e.BAD_MOVE)
            elif oldbombsight == (1,1,1,1):
                if or_tuples(oldsurroundings, oldbombdanger)[move_to_index[action]] == 0:  # walked in direction of no danger
                    if or_tuples(oldbombdanger, oldsurroundings)[oldtarget -1] == 0 and move_to_index[action] == oldtarget - 1:
                        events.append(e.PERFECT_MOVE)
                    elif or_tuples(oldbombdanger, oldsurroundings)[oldtarget -1] != 0 and move_to_index[action] != oldtarget - 1:
                        events.append(e.PERFECT_MOVE)
                    else:
                        events.append(e.BAD_MOVE)
                else:
                    events.append(e.BAD_MOVE)
            elif space == (1,1,1,1):
                if or_tuples(oldsurroundings, oldbombsight)[move_to_index[action]] == 0:  # walked in direction of no danger
                    if or_tuples(oldbombdanger, oldbombsight)[oldtarget -1] == 0 and move_to_index[action] == oldtarget - 1:
                        events.append(e.PERFECT_MOVE)
                    elif or_tuples(oldbombdanger, oldbombsight)[oldtarget -1] != 0 and move_to_index[action] != oldtarget - 1:
                        events.append(e.PERFECT_MOVE)
                    else:
                        events.append(e.BAD_MOVE)
                else:
                    events.append(e.BAD_MOVE)
            else:
                events.append(e.BAD_MOVE)
        else:  # everything else would be a bad decision
            events.append(e.BAD_MOVE)
    else:
        if action == 'WAIT':
            if oldbombdanger.count(0) == 0:  # if your surrounded or next step dangerous
                events.append(e.PERFECT_MOVE)
            elif oldfeatures[4] != 0 and oldfeatures[oldfeatures[4] - 1] == 1 and oldbombflag == 1:  # in front of target but cant lay bomb yet
                events.append(e.PERFECT_MOVE)
            elif oldtarget != 0 and oldbombdanger[oldtarget - 1] == 1: # target way is in danger so wait
                events.append(e.PERFECT_MOVE)
            elif np.count_nonzero(oldsurroundings) == 4:
                events.append(e.PERFECT_MOVE)
            else:
                events.append(e.BAD_MOVE)
        elif action == 'BOMB':
            if oldfeatures[4] != 0 and oldfeatures[oldtarget - 1] == 1:  # layed bomb in front of target
                if oldbombflag == 0:
                    events.append(e.PERFECT_MOVE)
                else:
                    events.append(e.BAD_MOVE)  # good moment but bombflag is 1
            else:
                events.append(e.BAD_MOVE)
        else:
            if or_tuples(oldbombdanger, oldsurroundings)[move_to_index[action]] == 1:  # walked against surrounding or into bomb
                events.append(e.BAD_MOVE)
            elif move_to_index[action] + 1 == oldtarget: # moved towards target
                events.append(e.PERFECT_MOVE)
            else:
                events.append(e.BAD_MOVE)
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

        old_features = self.current_features
        new_features = state_to_features(self, new_game_state)
        symmetry_features = generalize_feature(old_features, new_features, self_action)
        self.current_features = new_features

        other_events = []
        eventscopy = events.copy()
        for e in eventscopy:
            if e in delayed_events:
                events.remove(e)
                other_events.append(e)

        events = getArtificialRewards(old_features, new_features, self_action, events)
        eb = []
        # if old_features[10:14][old_features[4] - 1] == 0:
        """
        for a in ACTIONS:
            eb = getArtificialRewards(old_features, new_features, a, eb)
        if tuple(eb).count('PERFECT_MOVE') != 1:
            print(old_features)
        """
        reward = reward_from_events(self, events)

        # print(f"--------{old_game_state['step']}--{new_game_state['step']}-----------")
        # print(self.bomb)
        # print(f"Pos {old_game_state['self'][3][::-1]}---{new_game_state['self'][3][::-1]}")
        # print(old_features)
        # print(self.q_table[old_features])
        # print(self_action)
        # print(events)
        # print(reward)
        # print(new_features)
        # print(self.round_reward)
        # print("--------")

        # todo delayed
        if len(other_events) != 0:
            delayed_reward = reward_from_events(self, other_events)
            # print(self.transitions[-6])
            self.transitions[-6][1] += delayed_reward
            # print(self.transitions[-6])
            # print("------")


        transition = [symmetry_features, reward]
        self.transitions.append(transition)

        # print("------")
        self.round_reward += reward

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

    # todo last step rewards
    last_reward = reward_from_events(self, events)
    self.round_reward += last_reward
    last_symmetry_features = generalize_feature(self.current_features, self.current_features, last_action)

    self.all_rewards.append(self.round_reward)
    self.round_reward = 0
    self.round_scores.append(last_game_state['self'][1])
    mean_score = np.mean(self.round_scores[max(0, len(self.round_scores)-20):])
    self.avg_scores.append(mean_score)

    self.old_q_table = self.q_table.copy()

    for t in self.transitions:
        symmetry_features = t[0]
        reward = t[1]
        qfunction(self, symmetry_features, reward)

    for sym in last_symmetry_features:
        feat = tuple(sym[0][0])
        action = sym[1]
        self.q_table[feat][ACTIONS.index(action)] = last_reward

    self.transitions = []

    if mean_score > self.highest_score:
        self.highest_score = mean_score
        print(mean_score)
        with open("models/highest-new.pt", "wb") as file:
            pickle.dump(self.old_q_table, file)
        for k in range(5):
            print("YYYYY")

    if self.last_q_table is None:
        self.last_q_table = np.argmax(self.old_q_table, axis=14)
    else:
        newq = np.argmax(self.old_q_table, axis=14)
        changes = np.sum(self.last_q_table != newq)
        self.last_q_table = newq
        self.convergence.append(changes)

    self.survived_rounds.append(last_game_state['step'])
    #
    if (last_game_state['round'] % self.repetion) == 0:
        self.epsilon_start *= 0.5
    new_eps = (self.epsilon_start * (1 - (last_game_state['round'] % self.repetion) / self.repetion) ** 4)
    #new_eps = self.epsilon_start * (1 - (last_game_state['round'] / self.n_episodes))
    self.epsilon_cur = new_eps if new_eps > self.epsilon_min else self.epsilon_min
    self.epsilons.append(new_eps)
    # Store the models
    if last_game_state['round'] % SAVE_EVERY == 0 or last_game_state['round'] == self.n_episodes:
        with open("models/my-saved-models.pt", "wb") as file:
            pickle.dump(self.old_q_table, file)
        with open("progress.pt", "wb") as file:
            pickle.dump([self.all_rewards, self.round_scores, self.survived_rounds, self.convergence, self.epsilons], file)
        # with open("models/ftable.pt", "wb") as file:
        #     pickle.dump(self.f_table, file)
        # print(freq)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            # freq[event] += 1
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
