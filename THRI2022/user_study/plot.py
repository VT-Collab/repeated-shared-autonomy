import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

# plot for task 1
# Traverse all users, find task1 folder, collect all files
folder = "./user_data/"
ignored_folders = ["user0", "pilot"]
user_folders = glob.glob(folder + "user*")

noassist_lemon = []
casa_lemon = []
sari_lemon = []

noassist_soup = []
casa_soup = []
sari_soup = []

for user in user_folders:
    task_no = "Task1"
    task1_files = glob.glob(user + "/" + task_no + "/*")
    for filename in task1_files:
        data = pickle.load(open(filename, "rb"))
        filename = os.path.basename(filename)[:-4]
        method, task, demo_num = os.path.basename(filename).split("_")[:3]
        if method == "noassist" and task == "lemon":
            noassist_lemon.append(len(data))
        elif method == "casa" and task == "lemon":
            casa_lemon.append(len(data))
        elif method == "sari" and task == "lemon":
            sari_lemon.append(len(data))

        if method == "noassist" and task == "soup":
            noassist_soup.append(len(data))
        elif method == "casa" and task == "soup":
            casa_soup.append(len(data))
        elif method == "sari" and task == "soup":
            sari_soup.append(len(data))
    
print(sum(noassist_lemon), sum(casa_lemon), sum(sari_lemon))
print(sum(noassist_soup), sum(casa_soup), sum(sari_soup))
# d = pickle.load(open("./user_data/user1/Task1/casa_lemon_2_0.pkl", "rb"))
# print(d)