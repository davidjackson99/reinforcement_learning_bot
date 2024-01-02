import pickle
import matplotlib.pyplot as plt
rewards = []
import numpy as np

with open("progress.pt", "rb") as file:
    progress = pickle.load(file)

penalties = progress[0]
steps = progress[1]
score = progress[2]

x = np.arange(len(penalties))
plt.plot(x, penalties, label="penalty")
plt.plot(x, steps, label="Steps")
plt.legend()
plt.show()

plt.plot(x, score, label="Score")
plt.show()