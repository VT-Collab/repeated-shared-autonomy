import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():

    data_loc = 'data/test_runs/task3_noise/'

    total_ours_005 = []
    total_ours_007 = []
    total_ours_001 = []
    total_human_001 = []
    total_human_005 = []
    total_human_007 = []

    for filename in os.listdir(data_loc):
        noise = []

        traj = pickle.load(open(data_loc + "/" + filename, "rb"))
        if filename[2] == "o" and filename[16] == "1":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                noise.append(float(qdot_h * (1- confidence)))
            total_ours_001.append(noise)
        
        elif filename[2] == "o" and filename[16] == "7":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                noise.append(float(qdot_h * (1- confidence)))
            total_ours_007.append(noise)
        
        elif filename[2] == "o" and filename[16] == "5":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                noise.append(float(qdot_h * (1- confidence)))
            total_ours_005.append(noise)

        elif filename[2] == "h" and filename[16] == "1":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                noise.append(float(qdot_h * (1- confidence)))
            total_human_001.append(noise)
        
        elif filename[2] == "h" and filename[16] == "7":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                noise.append(float(qdot_h * (1- confidence)))
            total_human_007.append(noise)
        
        elif filename[2] == "h" and filename[16] == "5":
            for point in traj:
                confidence = point[-1]
                qdot_h = np.sum(np.abs(np.asarray(point[2])))
                noise.append(float(qdot_h * (1- confidence)))
            total_human_005.append(noise)

    ours = {}
    ours["001"] = total_ours_001
    ours["005"] = total_ours_005
    ours["007"] = total_ours_007

    human = {}
    human["001"] = total_human_001
    human["005"] = total_human_005
    human["007"] = total_human_007

    dataset = [ours, human]
    savename = "data/consolidated_data/task3_noise.pkl"
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[1]["001"])    
    # for i in range(5):
    # i = 0
    # x = np.arange(len(total_ours_005[i]))/len(total_ours_005[i])
    # plt.plot(x, total_ours_005[i], 'b', label = "ours")
    # x = np.arange(len(total_human_005[i]))/len(total_human_005[i])
    # plt.plot(x, total_human_005[i], 'g', label="human only")
    # # x = np.arange(len(total_noise_005[i]))/len(total_noise_005[i])
    # # plt.plot(x, total_noise_005[i], 'm', label="0.05")

    # plt.title("human effort while performing an unknown task (glass)")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
