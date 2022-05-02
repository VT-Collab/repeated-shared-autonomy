import time
import numpy as np
import pickle
import sys
import random
import os
import matplotlib.pyplot as plt

HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
SIGMA_D = np.identity(3) * 0.001

def plot_err():
    # get final state for each run from ours and ensemble per run
    # avg over runs and then over models for ours
    # plot
    goals = []
    for i in range(20):
        goal = np.array(pickle.load(open("goals/goals" + str(i+1) + ".pkl", "rb")))
        goals.append(goal)
    folder = "runs/ours/"
    ours_fse = {}
    ensemble_fse = {}
    for filename in os.listdir(folder):
        if filename[0] == ".":
            continue
        ours = pickle.load(open(folder + "/" + filename, "rb"))
        model = ours["model"].split("_")[1]
        run = ours["run"]
        task = ours["task"]
        model_num = int(ours["model"].split("_")[-1])
        state = np.array(ours["data"][-1][0][1])
        if not model in ours_fse:
            ours_fse[model] = np.zeros((int(model), 100))
        ours_fse[model][task-1, (model_num-1)*5+run-1] = np.linalg.norm(goals[int(model)-1][task-1] - state)
    for key in ours_fse:
        ours_fse[key] = [np.mean(ours_fse[key]), np.std(ours_fse[key])/np.sqrt(ours_fse[key].size)]
    
    folder = "runs/ensemble/"
    for filename in os.listdir(folder):
        if filename[0] == ".":
            continue
        ensemble = pickle.load(open(folder + "/" + filename, "rb"))
        model = ensemble["model"].split("_")[2]
        run = ensemble["run"]
        task = ensemble["task"]
        state = np.array(ensemble["data"][-1][0][1])
        if not model in ensemble_fse:
            ensemble_fse[model] = np.zeros((int(model), 100))
        ensemble_fse[model][task-1, run-1] = np.linalg.norm(goals[int(model)-1][task-1] - state)
    # print(ensemble_fse["1"])
    for key in ensemble_fse:
        # ensemble_fse[key] = np.mean(np.mean(ensemble_fse[key], axis=1))
        ensemble_fse[key] = [np.mean(ensemble_fse[key]), np.std(ensemble_fse[key])/np.sqrt(ensemble_fse[key].size)]



    ours_mean_err = []
    ensemble_mean_err = []
    ours_sem = []
    ensemble_sem = []
    for i in range(20):
        ours_mean_err.append(ours_fse[str(i+1)][0])
        ours_sem.append(ours_fse[str(i+1)][1])
        ensemble_mean_err.append(ensemble_fse[str(i+1)][0])
        ensemble_sem.append(ensemble_fse[str(i+1)][1])

    # plt.plot(np.arange(1,21),ours_mean_err)
    plt.plot(np.arange(1,21), ours_mean_err)
    plt.fill_between(np.arange(1,21), np.array(ours_mean_err)-np.array(ours_sem), np.array(ours_mean_err)+np.array(ours_sem))
    plt.errorbar(np.arange(1,21), ensemble_mean_err)
    plt.fill_between(np.arange(1,21), np.array(ensemble_mean_err)-np.array(ensemble_sem), np.array(ensemble_mean_err)+np.array(ensemble_sem))
    plt.show()
    # ensemble = np.array(ensemble["data"][-1][0][1]) 

    
    # print(np.linalg.norm(goal - ensemble))

def main():
    plot_err()
    


if __name__ == '__main__':
    main()