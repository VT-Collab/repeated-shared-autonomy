import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    dataset = []
    data_prefix = 'user/user'

    total_notepad_ours = []
    total_tape_ours = []
    total_tape_anca = []
    total_notepad_anca = []

    for usernumber in range(1,11):
        data_loc = data_prefix + str(usernumber) + '/runs'

        for filename in os.listdir(data_loc):
            data = 0.
            traj = pickle.load(open(data_loc + "/" + filename, "rb"))
            for point in traj:
                time = point[0]
                qdot_h = np.sum(np.abs(point[2]))
                beta = point[-1]
                if time >= 2.:
                    data += (1 - beta) * qdot_h

            if filename[0] == "r" and filename[4] == "n":
                total_notepad_ours.append(data)

            if filename[0] == "r" and filename[4] == "t":
                total_tape_ours.append(data)

        data_loc = data_prefix + str(usernumber) + '/policy_blending'

        for filename in os.listdir(data_loc):
            data = 0.
            traj = pickle.load(open(data_loc + "/" + filename, "rb"))
            for point in traj:
                time = point[0]
                qdot_h = np.sum(np.abs(point[2]))
                beta = point[-1]
                if time >= 2.:
                    data += (1 - beta) * qdot_h
            if filename[0] == "n":
                total_notepad_anca.append(data)

            if filename[0] == "t":
                total_tape_anca.append(data)


    dataset = {}
    max_notepad = np.max([np.mean(total_notepad_ours), np.mean(total_notepad_anca)])
    max_tape = np.max([np.mean(total_tape_ours), np.mean(total_tape_anca)])
    ours = [np.mean(total_notepad_ours)/max_notepad, np.mean(total_tape_ours)/max_tape]
    anca = [np.mean(total_notepad_anca)/max_notepad, np.mean(total_tape_anca)/max_tape]

    anca_sem = [np.std(total_notepad_anca)/(max_notepad * np.sqrt(10)), np.std(total_tape_anca)/(max_tape * np.sqrt(10))]
    ours_sem = [np.std(total_notepad_ours)/(max_notepad * np.sqrt(10)), np.std(total_tape_ours)/(max_tape * np.sqrt(10))]
    
    dataset["ours"] = ours
    dataset["anca"] = anca
    savename = 'data/consolidated_data/task1.pkl'
    pickle.dump(dataset, open(savename, "wb"))

    labels = ['Notepad', 'Tape']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, anca, width, yerr=anca_sem, label='Bayesian Inference')
    rects2 = ax.bar(x + width/2, ours, width, yerr=ours_sem, label='Ours')

    ax.set_ylabel('Human Effort')
    ax.set_title('Comparing human effort for known targets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()