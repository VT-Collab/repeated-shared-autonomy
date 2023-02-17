import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

# Task1_noassist = [1.6, 1.2, 6.95]
# Task1_sari = [6.8, 6.9, 5.15]
# Task1_casa = [3.9, 2.3, 2.7]

# Task2_noassist = [1.55, 1.65, 7]
# Task2_sari = [6.8, 6, 4.2]
# Task2_casa = [3.35, 2.2, 2.05]

# Task3_noassist = [1.3, 1.2]
# Task3_sari = [6.85, 6.5]
# Task3_casa = [3.05, 2.3]

# labels = ["recognize", "replicate", "return"]
# x = np.arange(len(labels))
# width = 0.35

# fig, ax = plt.subplots(1,3)
# rects1 = ax[0].bar(x - width/2, Task1_noassist, width/2, label="noassist")
# rects2 = ax[0].bar(x, Task1_casa, width/2, label="casa")
# rects2 = ax[0].bar(x + width/2, Task1_sari, width/2, label="ours")
# ax[0].set_xticks(x)
# ax[0].set_xticklabels(labels)
# ax[0].legend()

# rects1 = ax[1].bar(x - width/2, Task2_noassist, width/2, label="noassist")
# rects2 = ax[1].bar(x, Task2_casa, width/2, label="casa")
# rects2 = ax[1].bar(x + width/2, Task2_sari, width/2, label="ours")
# ax[1].set_xticks(x)
# ax[1].set_xticklabels(labels)
# ax[1].legend()

# labels = ["recognize", "replicate"]
# x = np.arange(len(labels))

# rects1 = ax[2].bar(x - width/2, Task3_noassist, width/2, label="noassist")
# rects2 = ax[2].bar(x, Task3_casa, width/2, label="casa")
# rects2 = ax[2].bar(x + width/2, Task3_sari, width/2, label="ours")
# ax[2].set_xticks(x)
# ax[2].set_xticklabels(labels)
# ax[2].legend()

# plt.show()

noassist_mean = [1.483333333, 1.35, 6.975, 0.1]
sari_mean = [6.816666667, 6.466666667, 4.675, 0.9]
casa_mean = [3.433333333, 2.266666667, 2.375, 0]

noassist_sem = [0.1290812092, 0.1058327216, 0.025, 0]
sari_sem = [0.05037523979, 0.09645310522, 0.2961018103, 0]
casa_sem = [0.2451411062, 0.1854398751, 0.2574218826, 0]

labels = ["recognize", "replicate", "return", "prefer"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, noassist_mean, width/2, yerr=noassist_sem, label="noassist")
rects2 = ax.bar(x, casa_mean, width/2, yerr=casa_sem, label="casa")
rects2 = ax.bar(x + width/2, sari_mean, width/2, yerr=sari_sem, label="ours")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig, ax = plt.subplots()
prefer = [0.1, 0, 0.9]
ax.bar(np.arange(3), prefer)
ax.set_ylim([0., 1.])

plt.show()