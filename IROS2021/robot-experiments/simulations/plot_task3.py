import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():

    data_loc = 'data/test_runs/task3/'

    total_ours_shelf = []
    total_human_only_shelf = []
    total_ours_glass = []
    total_human_only_glass = []

    for filename in os.listdir(data_loc):
        effort_ours = []
        human_only = []

        traj = pickle.load(open(data_loc + "/" + filename, "rb"))
        if filename[0] == "s" and filename[2] == "o":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                effort_ours.append(float(qdot_h * (1- confidence)))
            total_ours_shelf.append(effort_ours)

        if filename[0] == "g" and filename[2] == "o":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                effort_ours.append(float(qdot_h * (1- confidence)))
            total_ours_glass.append(effort_ours)

        elif filename[0] == "g" and filename[2] == "h":
            for point in traj:
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                human_only.append(float(qdot_h))
            total_human_only_glass.append(human_only)

        elif filename[0] == "s" and filename[2] == "h":
            for point in traj:
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                human_only.append(float(qdot_h))
            total_human_only_shelf.append(human_only)

    ours = {}
    ours["s"] = total_ours_shelf
    ours["g"] = total_ours_glass

    human = {}
    human["s"] = total_human_only_shelf
    human["g"] = total_human_only_glass

    dataset = [ours, human]
    savename = "data/consolidated_data/task3.pkl"
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0]["g"][20])

    # for i in range(5):
    #     x = np.arange(len(total_ours[i]))/len(total_ours[i])
    #     plt.plot(x, total_ours[i], 'b', label = "ours")
    #     x = np.arange(len(total_human_only[i]))/len(total_human_only[i])
    #     plt.plot(x, total_human_only[i], 'g', label="human only")

    # plt.title("human effort while performing an unknown task (glass)")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
