import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

# plot for task 1
# Traverse all users, find task1 folder, collect all files
folder = "./user_data/"
ignored_folders = ["user0", "pilot"]
user_folders = glob.glob(folder + "user*")

for user in user_folders:
    task_no = "Task1"
    task1_files = glob.glob(user + "/" + task_no + "/*")
    for file in task1_files:
        data = pickle.load(file)
