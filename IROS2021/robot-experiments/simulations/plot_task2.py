import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():

    dataset = []
    data_loc = 'data/test_runs/task2/'

    total_cnf_glass_ours = []
    total_cnf_glass_drop = []

    total_cnf_notepad_ours = []
    total_cnf_notepad_drop = []

    for filename in os.listdir(data_loc):
        cnf_drop = []
        cnf_ours = []

        traj = pickle.load(open(data_loc + "/" + filename, "rb"))
        if filename[0] == "g" and filename[2] == "o":
            for point in traj:
                time = point[0]
                cnf_ours.append(float(point[-1]))
            total_cnf_glass_ours.append(cnf_ours)

        elif filename[0] == "g" and filename[2] == "d":
            for point in traj:
                time = point[0]
                cnf_drop.append(float(point[-1]))           
            total_cnf_glass_drop.append(cnf_drop)

        elif filename[0] == "n" and filename[2] == "o":
            for point in traj:
                time = point[0]
                cnf_drop.append(float(point[-1]))           
            total_cnf_notepad_ours.append(cnf_drop)

        elif filename[0] == "n" and filename[2] == "d":
            for point in traj:
                time = point[0]
                cnf_drop.append(float(point[-1]))           
            total_cnf_notepad_drop.append(cnf_drop)

    dataset = {}

    ours = {}
    ours["g"] = total_cnf_glass_ours
    ours["n"] = total_cnf_notepad_ours

    dropout = {}
    dropout["g"] = total_cnf_glass_drop
    dropout["n"] = total_cnf_notepad_drop

    dataset = [ours, dropout]
    savename = 'data/consolidated_data/task2.pkl'
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[1]["n"][20])
    # for i in range(5):
    #     x = np.arange(len(total_cnf_ours[i]))/len(total_cnf_ours[i])
    #     plt.plot(x, total_cnf_ours[i], 'b', label = "ours")
    #     x = np.arange(len(total_cnf_drop[i]))/len(total_cnf_drop[i])
    #     plt.plot(x, total_cnf_drop[i], 'g', label="dropout")


    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
