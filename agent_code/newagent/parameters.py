import events as e
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 7  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

MEMORY_SIZE = 2000
BATCH_SIZE = 20
GAMMA = 0.95
LEARNING_RATE = 0.01

EPSILON_MAX = 0.95
EPSILON_MIN = 0.02

SAVE_EVERY = 5

game_rewards = {
    e.MOVED_LEFT: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_UP: -1,
    e.MOVED_RIGHT: -1,
    e.BOMB_DROPPED: -1,
    e.WAITED: -1,
    e.CRATE_DESTROYED: 30,
    e.COIN_FOUND: 60,
    e.COIN_COLLECTED: 100,
    e.KILLED_OPPONENT: 200,
    e.INVALID_ACTION: -50,
    e.KILLED_SELF: -300,
    e.RIGHT_DIRECTION: 10,
    e.WRONG_DIRECTION: -10,
    e.DODGED_BOMB:30,
    e.BOMB_STILL_IN_SIGHT:-10,
    e.PERFECT_BOMB: 50,
    e.SHOULD_BOMB:-50,
    e.MORE_DISTANCE_BOMB:15,
    e.LE_DISTANCE_BOMB:-20,
    'FINISHED_ROUND': 100,
    e.SURVIVED_ROUND:50,
    e.GOOD_WAIT:10,
    e.BAD_BOMB:-10
}

freq = {
    e.MOVED_LEFT: 0,
    e.MOVED_DOWN: 0,
    e.MOVED_UP: 0,
    e.MOVED_RIGHT: 0,
    e.BOMB_DROPPED: 0,
    e.WAITED: 0,
    e.CRATE_DESTROYED: 0,
    e.COIN_FOUND: 0,
    e.COIN_COLLECTED: 0,
    e.KILLED_OPPONENT: 0,
    e.INVALID_ACTION: 0,
    e.KILLED_SELF: 0,
    e.RIGHT_DIRECTION: 0,
    e.WRONG_DIRECTION: 0,
    e.DODGED_BOMB:0,
    e.BOMB_STILL_IN_SIGHT:0,
    e.PERFECT_BOMB: 0,
    e.MORE_DISTANCE_BOMB:0,
    e.LE_DISTANCE_BOMB:0,
    e.SHOULD_BOMB:0,
    e.GOOD_WAIT:0,
    'FINISHED_ROUND': 0,
    e.SURVIVED_ROUND:0,
    e.BAD_BOMB:0
}