import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("progress.pt", "rb") as file:
    progress = pickle.load(file)

rewards = progress[0]
scores = progress[1]
steps = progress[2]

x = np.arange(len(rewards))
plt.plot(x, rewards, label="Rewards")
plt.legend()
plt.show()

plt.plot(x, scores, label="Scores")
plt.legend()
plt.show()

plt.plot(x, steps, label="Steps")
plt.legend()
plt.show()