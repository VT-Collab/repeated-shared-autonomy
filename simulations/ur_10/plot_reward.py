import time
import numpy as np
import pickle
import sys
import random
import os
import matplotlib.pyplot as plt

def plot_reward():
    # get traj for each run from ours and ensemble per run and l2 norm with ideal traj
    # avg over runs and then over models for ours
    # plot
    tasklist = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
                      ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
                      ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
                      ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]
    
    folder = "runs/individual_models"
    optimal_traj = {}
    for filename in os.listdir(folder):
        if not filename[0] == "m":
            continue
        demo = pickle.load(open(folder + "/" + filename, "rb"), encoding='latin1')
        key = demo["task"]
        if key in optimal_traj:
            curr_traj = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
            optimal_traj[key] = np.append(optimal_traj[key], curr_traj, axis=0)
        else:
            optimal_traj[key] = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
    
    for key in optimal_traj:
        optimal_traj[key] = np.mean(optimal_traj[key], axis=0)

    folder = "runs/ours/"
    ours_fse = {}
    ensemble_fse = {}
    for filename in os.listdir(folder):
        if filename[0] == ".":
            continue
        ours = pickle.load(open(folder + "/" + filename, "rb"), encoding='latin1')
        model_data = ours["model"].split("_")
        model = model_data[:-1]
        run = ours["run"]
        task = ours["task"]
        if model_data[-1][-2] == "m":
            model_num = int(model_data[-1][-1])
        else:
            model_num = int(model_data[-1][-2:])
        ours_traj = np.array([item[1] for item in ours["data"]])
        ours_alpha = np.array([item[4] for item in ours["data"]])
        if not "_".join(model) in ours_fse:
            ours_fse["_".join(model)] = np.zeros((len(model), 100))
        # ours_fse["_".join(model)][model.index(task), (model_num)*5+run-1] = \
        #     np.sum(np.abs(np.linalg.norm(optimal_traj[task] - ours_traj)))
        ours_fse["_".join(model)][model.index(task), (model_num)*5+run-1] = \
            np.mean(ours_alpha)

    for key in ours_fse:
        ours_fse[key] = [np.mean(ours_fse[key]), np.std(ours_fse[key])/np.sqrt(ours_fse[key].size)]

    # for i in range(8):
    #     ours_costs = np.zeros(1,200)
    #     ours_costs[:, :100] = ours_fse["_".join(tasklist[i])]
    #     ours_costs[:, 100:] = ours_fse["_".join(tasklist[8+i])]
    
    folder = "runs/ensemble/"
    for filename in os.listdir(folder):
        if filename[0] == ".":
            continue
        ensemble = pickle.load(open(folder + "/" + filename, "rb"), encoding='latin1')
        model = ensemble["model"].split("_")
        run = ensemble["run"]
        task = ensemble["task"]
        ensemble_traj = np.array([item[1] for item in ensemble["data"]])
        ensemble_alpha = np.array([item[4] for item in ensemble["data"]])
        if not "_".join(model) in ensemble_fse:
            ensemble_fse["_".join(model)] = np.zeros((len(model), 100))
        # ensemble_fse["_".join(model)][model.index(task), run-1] = \
        # np.sum(np.abs(np.linalg.norm(optimal_traj[task] - ensemble_traj)))
        ensemble_fse["_".join(model)][model.index(task), run-1] = \
        np.mean(ensemble_alpha)
    # print(ensemble_fse["1"])
    for key in ensemble_fse:
        # ensemble_fse[key] = np.mean(np.mean(ensemble_fse[key], axis=1))
        ensemble_fse[key] = [np.mean(ensemble_fse[key]), np.std(ensemble_fse[key])/np.sqrt(ensemble_fse[key].size)]



    ours_mean_err = []
    ensemble_mean_err = []
    # ours_sem = []
    # ensemble_sem = []
    for tasks in tasklist:
        ours_mean_err.append(ours_fse["_".join(tasks)][0])
        # ours_sem.append(ours_fse["_".join(tasks)][1])
        ensemble_mean_err.append(ensemble_fse["_".join(tasks)][0])
        # ensemble_sem.append(ensemble_fse["_".join(tasks)][1])
    ours_mean_err = np.vstack((ours_mean_err[:8], ours_mean_err[8:]))
    ours_sem = np.std(ours_mean_err, axis=0) / np.sqrt(2)
    ours_mean_err = np.mean(ours_mean_err, axis=0)
    ensemble_mean_err = np.vstack((ensemble_mean_err[:8], ensemble_mean_err[8:]))
    ensemble_sem = np.std(ensemble_mean_err, axis=0) / np.sqrt(2)
    ensemble_mean_err = np.mean(ensemble_mean_err, axis=0)
    plt.errorbar(np.arange(1,len(ensemble_mean_err)+1),ensemble_mean_err, yerr=ensemble_sem, label="ensemble")
    plt.errorbar(np.arange(1,len(ours_mean_err)+1),ours_mean_err, yerr=ours_sem, label="Ours")
    plt.xlabel('# skills learned')
    plt.ylabel('Mean cost')
    plt.legend()
    # plt.plot(np.arange(1,21), ours_mean_err)
    # plt.fill_between(np.arange(1,9), np.array(ours_mean_err)-np.array(ours_sem), np.array(ours_mean_err)+np.array(ours_sem))
    # plt.fill_between(np.arange(1,9), np.array(ensemble_mean_err)-np.array(ensemble_sem), np.array(ensemble_mean_err)+np.array(ensemble_sem))
    plt.show()
    # ensemble = np.array(ensemble["data"][-1][0][1]) 

    
    # print(np.linalg.norm(goal - ensemble))

def main():
    plot_reward()
    


if __name__ == '__main__':
    main()