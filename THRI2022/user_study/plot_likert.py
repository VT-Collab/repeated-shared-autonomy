import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

Task1_noassist = [1.6, 1.2, 6.95]
Task1_sari = [6.8, 6.9, 5.15]
Task1_casa = [3.9, 2.3, 2.7]

Task2_noassist = [1.55, 1.65, 7]
Task2_sari = [6.8, 6, 4.2]
Task2_casa = [3.35, 2.2, 2.05]

Task3_noassist = [1.3, 1.2]
Task3_sari = [6.85, 6.5]
Task3_casa = [3.05, 2.3]

labels = ["recognize", "replicate", "return"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(1,3)
rects1 = ax[0].bar(x - width/2, Task1_noassist, width/2, label="noassist")
rects2 = ax[0].bar(x, Task1_casa, width/2, label="casa")
rects2 = ax[0].bar(x + width/2, Task1_sari, width/2, label="ours")
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].legend()

rects1 = ax[1].bar(x - width/2, Task2_noassist, width/2, label="noassist")
rects2 = ax[1].bar(x, Task2_casa, width/2, label="casa")
rects2 = ax[1].bar(x + width/2, Task2_sari, width/2, label="ours")
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].legend()

labels = ["recognize", "replicate"]
x = np.arange(len(labels))

rects1 = ax[2].bar(x - width/2, Task3_noassist, width/2, label="noassist")
rects2 = ax[2].bar(x, Task3_casa, width/2, label="casa")
rects2 = ax[2].bar(x + width/2, Task3_sari, width/2, label="ours")
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels)
ax[2].legend()

plt.show()