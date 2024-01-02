import events as e
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 7  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

LEARNING_RATE = 0.6
DISCOUNT = 0.95

game_rewards = {
    e.COIN_COLLECTED: 100,
    e.KILLED_OPPONENT: 5,
    e.INVALID_ACTION: -50,
    e.MOVED_LEFT: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_UP: -1,
    e.MOVED_RIGHT: -1,
    e.WAITED: -1,
    e.BOMB_DROPPED: 5,
    e.CRATE_DESTROYED: 10,
    e.KILLED_SELF: -50,
    "INBOMBRADIUS": -10,
    "GOODPLACEMENT": 10,
    "BADPLACEMENT": -10,
    "NOBOMB": -10,
    "MOREBOMBDIS": 15,
    "LESSBOMBDIS": -15,
    "ESCAPED": 40,
    "BACKIN": -15,
    "BOMBPUSH": -10

}