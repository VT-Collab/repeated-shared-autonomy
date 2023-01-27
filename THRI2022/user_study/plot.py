import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

# plot for task 1
# Traverse all users, find task1 folder, collect all files
folder = "./user_data/"
user_folders = glob.glob(folder + "user*")

tasks = ["Task1", "Task2", "Task3"]
methods = ["noassist", "casa", "sari"]

# for each metric
time_known = {}
action_time_known = {}
fighting_time_known = {}
time_unknown = {}
for method in methods:
    # n_users x n_demos x n_tasks
    time_known[method] = np.zeros((len(user_folders), 3, 3))
    action_time_known[method] = np.zeros((len(user_folders), 3, 3))
    fighting_time_known[method] = np.zeros((len(user_folders), 3, 3))
    time_unknown[method] = np.zeros((len(user_folders), 3, 2))

for user_num, user in enumerate(user_folders):
    for task_no, task_fname in enumerate(tasks):
        filenames = glob.glob(user + "/" + task_fname + "/*")
        for method in methods:
            method_filenames = [filename for filename in filenames if method in filename]
            if method == "noassist":
                key_human = "xdot_h"
                key_robot = ""
            else:
                key_human = "a_human"
                key_robot = "a_robot"
            for filename in method_filenames:
                data = pickle.load(open(filename, "rb"))  
                filename = os.path.basename(filename)[:-4]
                _, task, demo_num = os.path.basename(filename).split("_")[:3]
                # convert from 1 index to 0 index
                demo_num = int(demo_num) - 1
                # find known task per method
                if task_no == 0: 
                    if task == "lemon":
                        time_known[method][user_num, int(demo_num), 0] = len(data)
                        action_timesteps = 0
                        fighting_timesteps = 0
                        for item in data:
                            if np.sum(np.abs(item[key_human])) > 0:
                                action_timesteps += 1
                                if key_robot and np.dot(item[key_human], item[key_robot]) < 0:
                                    fighting_timesteps += 1
                        action_time_known[method][user_num, int(demo_num), 0] = float(action_timesteps) / len(data)
                        fighting_time_known[method][user_num, int(demo_num), 0] = float(fighting_timesteps) / len(data)
                
                    elif task == "soup":
                        time_unknown[method][user_num, int(demo_num), 0] = len(data)
                
                elif task_no == 1: 
                    if task == "soup":
                        time_known[method][user_num, int(demo_num), 1] = len(data)
                        action_timesteps = 0
                        fighting_timesteps = 0
                        for item in data:
                            if np.sum(np.abs(item[key_human])) > 0:
                                action_timesteps += 1
                                if key_robot and np.dot(item[key_human], item[key_robot]) < 0:
                                    fighting_timesteps += 1
                        action_time_known[method][user_num, int(demo_num), 1] = float(action_timesteps) / len(data)
                        fighting_time_known[method][user_num, int(demo_num), 1] = float(fighting_timesteps) / len(data)
                
                    elif task == "stir":
                        time_unknown[method][user_num, int(demo_num), 1] = len(data)
                
                elif task_no == 2 and task == "stir":
                    time_known[method][user_num, int(demo_num), 2] = len(data)
                    action_timesteps = 0
                    fighting_timesteps = 0
                    for item in data:
                        if np.sum(np.abs(item[key_human])) > 0:
                            action_timesteps += 1
                            if key_robot and np.dot(item[key_human], item[key_robot]) < 0:
                                    fighting_timesteps += 1
                    action_time_known[method][user_num, int(demo_num), 2] = float(action_timesteps) / len(data)
                    fighting_time_known[method][user_num, int(demo_num), 2] = float(fighting_timesteps) / len(data)

labels = ["lemon", "soup", "stir"]
noassist_means = []
casa_means = []
sari_means = []
for idx in range(3):
    noassist_means.append(np.mean(time_known["noassist"][:,:,idx]))
    casa_means.append(np.mean(time_known["casa"][:,:,idx]))
    sari_means.append(np.mean(time_known["sari"][:,:,idx]))

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, noassist_means, width/2, label="noassist")
rects2 = ax.bar(x, casa_means, width/2, label="casa")
rects2 = ax.bar(x + width/2, sari_means, width/2, label="ours")

ax.set_ylabel("Total timesteps")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
# fig.tight_layout()
noassist_means = []
casa_means = []
sari_means = []
for idx in range(3):
    noassist_means.append(np.mean(action_time_known["noassist"][:,:,idx]))
    casa_means.append(np.mean(action_time_known["casa"][:,:,idx]))
    sari_means.append(np.mean(action_time_known["sari"][:,:,idx]))

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, noassist_means, width/2, label="noassist")
rects2 = ax.bar(x, casa_means, width/2, label="casa")
rects2 = ax.bar(x + width/2, sari_means, width/2, label="ours")

ax.set_ylabel("Action timesteps")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

noassist_means = []
casa_means = []
sari_means = []
for idx in range(3):
    noassist_means.append(np.mean(fighting_time_known["noassist"][:,:,idx]))
    casa_means.append(np.mean(fighting_time_known["casa"][:,:,idx]))
    sari_means.append(np.mean(fighting_time_known["sari"][:,:,idx]))

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, noassist_means, width/2, label="noassist")
rects2 = ax.bar(x, casa_means, width/2, label="casa")
rects2 = ax.bar(x + width/2, sari_means, width/2, label="ours")

ax.set_ylabel("Action timesteps")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()



plt.show()
