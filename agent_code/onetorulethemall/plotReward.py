import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("progress.pt", "rb") as file:
    progress = pickle.load(file)
rewards = progress[0]
scores = progress[1]
steps = progress[2]
convergence = progress[3]
epsilon = progress[4]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

rewards = moving_average(rewards, 20)
scores = moving_average(scores, 20)
steps = moving_average(steps, 20)



x = np.arange(len(rewards))
plt.plot(x, rewards, label="Rewards")
plt.legend()
plt.grid()
plt.show()

plt.plot(x, scores, label="Scores")
plt.plot(np.arange(len(epsilon)), epsilon, label="Epsilon")
plt.legend()
plt.grid()
plt.show()

plt.plot(x, steps, label="Steps")
plt.legend()
plt.grid()
plt.show()

x = np.arange(len(convergence))
plt.plot(x, convergence, label="Convergence")
plt.legend()
plt.grid()
plt.show()

# x = np.arange(len(epsilon))
# plt.plot(x, epsilon, label="Epsilon")
# plt.legend()
# plt.grid()
# plt.show()

# with open("models/ftable.pt", "rb") as file:
#     ftable = pickle.load(file)
#
# print(np.mean(ftable))