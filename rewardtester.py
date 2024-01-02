import numpy as np
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
from agent_code.onetorulethemall.train import getArtificialRewards

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

# for k in range(10):
#     s = np.random.randint(2, size=4)
#     t = np.random.randint(1, 5, size=1)
#     bf = np.random.randint(2, size=1)
#     bs = np.random.randint(2, size=4)
#     bd = np.random.randint(2, size=4)
#     a = np.random.choice(ACTIONS)

# whole = tuple(np.hstack((s, t, bf, bs, bd)))
whole = (1,0,1,1,4,1,0,0,0,1,0,1,0,0)
print(whole)
e = []
for a in ACTIONS:
    e = getArtificialRewards(whole, whole, a, e)
print(e)

# a = 'DOWN'
# print(tuple(s), tuple(t), tuple(bf), tuple(bs), tuple(bd), a, "->", getArtificialRewards(whole, whole, a))
# print(whole, a, "->", getArtificialRewards(whole, whole, a))

