import pickle
import matplotlib.pyplot as plt
rewards = []
import numpy as np

with open("all-rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    print(rewards)

plt.plot(np.arange(len(rewards)), rewards)
plt.show()