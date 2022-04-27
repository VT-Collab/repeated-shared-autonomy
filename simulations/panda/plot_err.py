import time
import numpy as np
import pickle
import sys
import random
import os


HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
SIGMA_D = np.identity(3) * 0.001

def plot_err():
    # get final state for each run from ours and ensemble per run
    # avg over runs and then over models for ours
    # plot
    goal = np.array(pickle.load(open("goals/goals1.pkl", "rb")))
    folder = "runs/ours/"
    ours_1_1 = []
    for filename in os.listdir(folder):
        ours = pickle.load(open(folder + "/" + filename, "rb"))
        if ours["model"][6:8] ==  "1_" and ours["task"] == 1:
            print(ours["model"])
            ours = np.array(ours["data"][-1][0][1]) 
            ours_1_1.append(np.linalg.norm(goal - ours))
    # ensemble = pickle.load(open("runs/ensemble/model_ensemble_1 _task_1_run_1.pkl", "rb"))
    # print(len(ours_1_1))
    
    # ensemble = np.array(ensemble["data"][-1][0][1]) 

    
    # print(np.linalg.norm(goal - ensemble))

def main():
    plot_err()
    


if __name__ == '__main__':
    main()