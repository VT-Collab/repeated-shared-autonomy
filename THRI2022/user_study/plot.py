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
time_unknown = {}
for method in methods:
    # n_users x n_demos x n_tasks
    time_known[method] = np.zeros((len(user_folders), 3, 3))
    action_time_known[method] = np.zeros((len(user_folders), 3, 3))
    time_unknown[method] = np.zeros((len(user_folders), 3, 2))

for user_num, user in enumerate(user_folders):
    for task_no, task_fname in enumerate(tasks):
        filenames = glob.glob(user + "/" + task_fname + "/*")
        for method in methods:
            method_filenames = [filename for filename in filenames if method in filename]
            if method == "noassist":
                key = "xdot_h"
            else:
                key = "a_human"
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
                        for item in data:
                            if np.sum(item[key]) > 0:
                                action_timesteps += 1
                        action_time_known[method][user_num, int(demo_num), 0] = action_timesteps
                    elif task == "soup":
                        time_unknown[method][user_num, int(demo_num), 0] = len(data)
                elif task_no == 1: 
                    if task == "soup":
                        time_known[method][user_num, int(demo_num), 1] = len(data)
                        action_timesteps = 0
                        for item in data:
                            if np.sum(item[key]) > 0:
                                action_timesteps += 1
                        action_time_known[method][user_num, int(demo_num), 1] = action_timesteps
                    elif task == "stir":
                        time_unknown[method][user_num, int(demo_num), 1] = len(data)
                elif task_no == 2 and task == "stir":
                    time_known[method][user_num, int(demo_num), 2] = len(data)
                    action_timesteps = 0
                    for item in data:
                        if np.sum(item[key]) > 0:
                            action_timesteps += 1
                    action_time_known[method][user_num, int(demo_num), 2] = action_timesteps

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
plt.show()
# plt.plot()
# print(time_known["casa"][:,:,0])
# print(time_known["sari"][:,:,0])
        # for filename in task1_files:
        #     data = pickle.load(open(filename, "rb"))
        #     filename = os.path.basename(filename)[:-4]
        #     method, task, demo_num = os.path.basename(filename).split("_")[:3]
        #     if method == "noassist" and task == "lemon":
        #         noassist_lemon.append(len(data))
        #     elif method == "casa" and task == "lemon":
        #         casa_lemon.append(len(data))
        #     elif method == "sari" and task == "lemon":
        #         sari_lemon.append(len(data))

        #     if method == "noassist" and task == "soup":
        #         noassist_soup.append(len(data))
        #     elif method == "casa" and task == "soup":
        #         casa_soup.append(len(data))
        #     elif method == "sari" and task == "soup":
        #         sari_soup.append(len(data))
    
# print(sum(noassist_lemon), sum(casa_lemon), sum(sari_lemon))
# print(sum(noassist_soup), sum(casa_soup), sum(sari_soup))
# d = pickle.load(open("./user_data/user1/Task1/casa_lemon_2_0.pkl", "rb"))
# print(d)