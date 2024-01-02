import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import look_for_targets, ACTIONS, state_to_features
import matplotlib.pyplot as plt
from .parameters import TRANSITION_HISTORY_SIZE, RECORD_ENEMY_TRANSITIONS, LEARNING_RATE, DISCOUNT, game_rewards, SAVE_EVERY
import collections

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
COIN_LESSDISTANCE_EVENT = 'COIN_LESSDISTANCE'
COIN_SAMEDISTANCE_EVENT = 'COIN_SAMEDISTANCE'
COIN_MOREDISTANCE_EVENT = 'COIN_MOREDISTANCE'
CRATE_LESSDISTANCE_EVENT = 'CRATE_LESSDISTANCE_EVENT'
CRATE_SAMEDISTANCE_EVENT = 'CRATE_SAMEDISTANCE_EVENT'
CRATE_MOREDISTANCE_EVENT = 'CRATE_MOREDISTANCE_EVENT'
BOMB_LESSDISTANCE_EVENT = 'BOMB_LESSDISTANCE_EVENT'
BOMB_SAMEDISTANCE_EVENT = 'BOMB_SAMEDISTANCE_EVENT'
BOMB_MOREDISTANCE_EVENT = 'BOMB_MOREDISTANCE_EVENT'
BOMB_MOREDISTANCE_EVENT = 'BOMB_MOREDISTANCE_EVENT'
BOMB_SAVING_EVENT = 'BOMB_SAVING_EVENT'
BOMB_RISK_EVENT = 'BOMB_RISK_EVENT'
DEADEND_BOMB = 'DEADEND_BOMB'
FREESPACE_BOMB = 'FREESPACE_BOMB'
WRONG_WAIT = 'WRONG_WAIT'
BOMB_DROPPED_WRONGTIME = 'BOMB_DROPPED_WRONGTIME'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.all_rewards = []
    self.round_reward = 0
    self.highest_reward = -10000000
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)



def eval_move(self, oldpos, newpos, otype):
    if otype == "COIN":
        target = self.target_coin
    elif otype == "CRATE":
        target = self.target_crate
    elif otype == "BOMB":
        target = self.target_bomb
    else:
        target = self.target_deadend
    diff = (newpos[0]- oldpos[0]+1, newpos[1]- oldpos[1]+1)
    if diff == (1,1):
        if otype == "COIN":
            return COIN_SAMEDISTANCE_EVENT
        elif otype == "CRATE":
            return CRATE_SAMEDISTANCE_EVENT
        elif otype == "BOMB":
            return BOMB_SAMEDISTANCE_EVENT
    elif diff == target:
        if otype == "COIN":
            return COIN_LESSDISTANCE_EVENT
        elif otype == "CRATE":
            return CRATE_LESSDISTANCE_EVENT
        elif otype == "BOMB":
            return BOMB_LESSDISTANCE_EVENT
    else:
        if otype == "COIN":
            return COIN_MOREDISTANCE_EVENT
        elif otype == "CRATE":
            return CRATE_MOREDISTANCE_EVENT
        elif otype == "BOMB":
            return BOMB_MOREDISTANCE_EVENT



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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None:
        # Idea: Add your own events to hand out rewards
        # state_to_features is defined in callbacks.py

        old_features = self.current_features


        if self.bomb_ticker > 1: #bombe liegt im spiel

            bombs = new_game_state['bombs']
            bomb_xys = [xy for (xy, t) in bombs]
            try:
                bomb_pos = bomb_xys[0]
                self.prev_pos = bomb_pos
            except:
                bomb_pos = self.prev_pos #wenn bombe in diesem moment explodiert

            if e.BOMB_EXPLODED in events:
                events.append(BOMB_SAVING_EVENT)  # Bedeutet agent hat überlebt
                bombs = new_game_state['bombs']
                bomb_xys = [xy for (xy, t) in bombs]
                # only delete target bomb if this specific bomb exploded
                if self.target_bomb not in bomb_xys:
                    self.target_bomb = None

            if self_action == 'WAIT':
                if new_game_state['self'][3][0] == bomb_pos[0] or new_game_state['self'][3][1] == bomb_pos[1]:
                    events.append(eval_move(self, old_game_state['self'][3], new_game_state['self'][3], "BOMB"))
                    events.append(WRONG_WAIT)

            else:
                events.append(eval_move(self, old_game_state['self'][3], new_game_state['self'][3], "BOMB"))

                if old_game_state['self'][3][0] == bomb_pos[0] or old_game_state['self'][3][1] == bomb_pos[1]:
                    if new_game_state['self'][3][0] != bomb_pos[0] and new_game_state['self'][3][1] != bomb_pos[1]:
                        events.append(BOMB_SAVING_EVENT)
                else:
                    # see if agent took step that will get himself killed
                    if new_game_state['self'][3][0] == bomb_pos[0] or new_game_state['self'][3][1] == bomb_pos[1]:
                        events.append(BOMB_RISK_EVENT)

                events.append(eval_move(self, old_game_state['self'][3], new_game_state['self'][3], "BOMB"))

        elif self.target_coin != None: #Keine bombe aber target coin im spiel
            events.append(eval_move(self, old_game_state['self'][3], new_game_state['self'][3], "COIN"))

            if self_action == 'BOMB':
                events.append(BOMB_DROPPED_WRONGTIME)
                new_features = state_to_features(self, new_game_state)
                if new_features[0] == 1 or new_features[1] == 1 or new_features[2] == 1 or new_features[3] == 1:
                    events.append(DEADEND_BOMB)
                else:
                    events.append(FREESPACE_BOMB)

        elif self.target_crate != None: #keine coins & bomben im spiel, nur crates
            events.append(eval_move(self, old_game_state['self'][3], new_game_state['self'][3], "CRATE"))

            if self_action == 'WAIT':
                events.append(e.WAITED)

            if self_action == 'BOMB':
                events.append(e.BOMB_DROPPED)
                new_features = state_to_features(self, new_game_state)
                if new_features[0] == 1 or new_features[1] == 1 or new_features[2] == 1 or new_features[3] == 1:
                    events.append(DEADEND_BOMB)
                else:
                    events.append(FREESPACE_BOMB)

        if e.CRATE_DESTROYED in events:
            # only delete target crate if this specific coin was destroyed
            field = new_game_state['field']
            crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (field[x, y] == 1)]
            if self.target_crate not in crates:
                self.target_crate = None

        if e.COIN_COLLECTED in events:
            # only delete target coin if this specific coin was collected
            if self.target_coin not in new_game_state['coins']:
                self.target_coin = None

        new_features = state_to_features(self, new_game_state)
        self.current_features = new_features

        reward = reward_from_events(self, events)
        # for k in events:
        #     print(k)
        # print(e.WAITED)
        # print("--------")
        transition = Transition(old_features, self_action, new_features, reward)

        # if e.WAITED in events:
        #     print("")
        if e.COIN_COLLECTED in events:
            self.q_table[old_features][ACTIONS.index(self_action)] = game_rewards[e.COIN_COLLECTED]
        if e.CRATE_DESTROYED in events:
            self.q_table[old_features][ACTIONS.index(self_action)] = game_rewards[e.CRATE_DESTROYED]
        else:
            max_future_q = np.max(self.q_table[new_features])
            current_q = self.q_table[old_features][ACTIONS.index(self_action)]

            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
            self.q_table[old_features][ACTIONS.index(self_action)] = new_q
        self.round_reward += reward

        self.transitions.append(transition)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #print(self.round_reward, "coins: ", last_game_state['self'][1])
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    r = reward_from_events(self, events)
    self.round_reward += r
    self.transitions.append(
        Transition(state_to_features(self, last_game_state), last_action, None, r))
    last_game_features = state_to_features(self, last_game_state)
    if e.GOT_KILLED in events:
        self.q_table[last_game_features][ACTIONS.index(last_action)] = game_rewards[
            e.GOT_KILLED]
    if e.KILLED_SELF in events:
        self.q_table[last_game_features][ACTIONS.index(last_action)] = game_rewards[
            e.KILLED_SELF]
    elif e.SURVIVED_ROUND in events:
        self.q_table[last_game_features][ACTIONS.index(last_action)] = game_rewards[
            e.SURVIVED_ROUND]


    self.random_prob = self.random_prob * self.random_decay
    round = last_game_state['round']
    coins = last_game_state['self'][1]
    self.all_rewards.append(self.round_reward)
    #print(self.round_reward, "Coins", len(last_game_state['coins']))
    self.logger.debug(f'Round {round}  --> Reward: {self.round_reward}, Coins: {coins}')
    self.current_features = None
    self.round_reward = 0
    # Store the models

    if round == self.n_episodes or last_game_state['round'] % SAVE_EVERY == 0:
        plt.plot(list(range(len(self.all_rewards))), self.all_rewards)
        plt.show()
        with open("all-rewards.pt", "wb") as file:
            pickle.dump(self.all_rewards, file)
        with open(f"models/my-saved-model-{last_game_state['round']}.pt", "wb") as file:
            pickle.dump(self.q_table, file)


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
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

