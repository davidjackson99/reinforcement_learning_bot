import events as e
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 7  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

LEARNING_RATE = 0.6
DISCOUNT = 0.95

SHOW_EVERY = 50

game_rewards = {
    e.COIN_COLLECTED: 90,
    e.COIN_FOUND : 20,
    e.KILLED_OPPONENT: 100,
    e.INVALID_ACTION: -20,
    e.MOVED_LEFT: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_UP: -1,
    e.MOVED_RIGHT: -1,
    e.WAITED: -1,
    e.BOMB_DROPPED: -1,
    e.CRATE_DESTROYED: 50,
    e.KILLED_SELF: -300,
    e.GOT_KILLED: -200,
    e.SURVIVED_ROUND: 500
}
