import events as e
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 7  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

LEARNING_RATE = 0.2
DISCOUNT = 0.5

SAVE_EVERY = 200

game_rewards = {
    e.MOVED_LEFT: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_UP: -1,
    e.MOVED_RIGHT: -1,
    e.BOMB_DROPPED: -1,
    e.WAITED: -1,
    e.CRATE_DESTROYED: 10,
    e.COIN_COLLECTED: 30,
    e.KILLED_OPPONENT: 100,
    e.INVALID_ACTION: -50,
    e.KILLED_SELF: -20,
    e.PERFECT_MOVE: 50,
    'FINISHED_ROUND': 100,
    e.BAD_MOVE:-50,
}

delayed_events = [
e.CRATE_DESTROYED,
e.KILLED_OPPONENT,
]
