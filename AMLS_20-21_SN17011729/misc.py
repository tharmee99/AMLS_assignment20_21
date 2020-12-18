from matplotlib import pyplot as plt
import numpy as np

a1 = [[468, 33],
      [32, 467]]
a2 = [[449, 51],
      [87, 413]]
b1 = [[484, 10, 0, 5, 1],
      [4, 492, 0, 2, 2],
      [0, 0, 500, 0, 0],
      [2, 7, 0, 490, 1],
      [2, 0, 0, 0, 498]]
b2 = [[400, 2, 4, 3, 97],
      [11, 377, 5, 6, 84],
      [14, 1, 424, 4, 82],
      [10, 4, 3, 403, 94],
      [17, 0, 1, 0, 454]]

genders = ['Female', 'Male']
emotions = ['Not Smiling', 'Smiling']
eye_colours = ['Brown', 'Blue', 'Green', 'Grey', 'Black']

fig, ax = plt.subplots()
im = ax.imshow(b1, cmap='gist_yarg')

for i in range(5):
    for j in range(5):
        text = ax.text(j, i, b1[i][j], ha="center", va="center", color="r", fontsize=16)

ax.xaxis.tick_top()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

###########################################################################################

fig, ax = plt.subplots()
im = ax.imshow(b2, cmap='gist_yarg')

for i in range(5):
    for j in range(5):
        text = ax.text(j, i, b2[i][j], ha="center", va="center", color="r", fontsize=16)

ax.set_xticks(np.arange(len(eye_colours)))
ax.set_yticks(np.arange(len(eye_colours)))

ax.set_xticklabels(eye_colours)
ax.set_yticklabels(eye_colours)

ax.xaxis.tick_top()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

##########################################################################################

fig, ax = plt.subplots()
im = ax.imshow(a1, cmap='gist_yarg')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, a1[i][j], ha="center", va="center", color="r", fontsize=16)

ax.set_xticks(np.arange(len(genders)))
ax.set_yticks(np.arange(len(genders)))

ax.set_xticklabels(genders)
ax.set_yticklabels(genders)

ax.xaxis.tick_top()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

##########################################################################################

fig, ax = plt.subplots()
im = ax.imshow(a2, cmap='gist_yarg')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, a2[i][j], ha="center", va="center", color="r", fontsize=16)

ax.set_xticks(np.arange(len(emotions)))
ax.set_yticks(np.arange(len(emotions)))

ax.set_xticklabels(emotions)
ax.set_yticklabels(emotions)

ax.xaxis.tick_top()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()