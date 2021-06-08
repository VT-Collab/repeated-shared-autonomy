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

    cup_ours_6 = []
    shelf_ours_6 = []
    cup_ours_3 = []
    cup_none = []
    shelf_none = []
    
    for usernumber in range(1,11):
        data_loc = data_prefix + str(usernumber) + '/runs'

        for filename in os.listdir(data_loc):
            data = 0.
            traj = pickle.load(open(data_loc + "/" + filename, "rb"))
            n_states = len(traj)
            lookahead = 5
            for idx in range(n_states-lookahead):
                start = np.asarray(traj[idx][1])
                end = np.asarray(traj[idx + lookahead][1])
                b_1 = traj[idx][-1]
                b_2 = traj[idx + lookahead][-1]
                qdot_h = np.sum(np.abs(end - start))
                beta = (b_1 + b_2)/2.
                data += (1 - beta) * qdot_h
                
            if filename[0] == "r" and filename[4] == "u":
                cup_ours_6.append(data)

            if filename[0] == "r" and filename[4] == "s":
                shelf_ours_6.append(data)

            if filename[0] == "u":
                cup_ours_3.append(data)

        data_loc = data_prefix + str(usernumber) + '/demos'
        for filename in os.listdir(data_loc):
            data = 0.
            traj = pickle.load(open(data_loc + "/" + filename, "rb"))
            n_states = len(traj)
            lookahead = 5
            for idx in range(n_states-lookahead):
                start = np.asarray(traj[idx][7:])
                end = np.asarray(traj[idx + lookahead][7:])
                qdot_h = np.sum(np.abs(end - start))
                data += qdot_h

            if filename[0] == "s":
                shelf_none.append(data)

            if filename[0] == "u":
                cup_none.append(data)

    ours = {}
    ours["cup"] = np.mean(cup_ours_6)
    ours["shelf"] = np.mean(shelf_ours_6)
    none = {}
    none["cup"] = np.mean(cup_none)
    none["shelf"] = np.mean(shelf_none)
    dataset = [ours, none]
    savename = 'data/consolidated_data/task2.pkl'
    pickle.dump(dataset, open(savename, "wb"))

    max_cup = np.max([np.mean(cup_ours_6), np.mean(cup_ours_3), np.mean(cup_none)])
    max_shelf = np.max([np.mean(shelf_ours_6), np.mean(shelf_none)])
    ours_6 = [np.mean(cup_ours_6)/max_cup, np.mean(shelf_ours_6)/max_shelf]
    ours_3 = [np.mean(cup_ours_3)/max_cup, 0]
    none = [np.mean(cup_none)/max_cup, np.mean(shelf_none)/max_shelf]

    ours_6_sem = [np.std(cup_ours_6)/(max_cup*np.sqrt(10)), np.std(shelf_ours_6)/(max_shelf*np.sqrt(10))]
    ours_3_sem = [np.std(cup_ours_3)/(max_cup*np.sqrt(10)), 0]
    none_sem = [np.std(cup_none)/(max_cup*np.sqrt(10)), np.std(shelf_none)/(max_shelf*np.sqrt(10))]

    labels = ['Cup', 'Shelf']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, none, width, yerr=none_sem, label='No Assist')
    rects2 = ax.bar(x + width, ours_3, width, yerr=ours_3_sem, label='Ours-3')
    rects3 = ax.bar(x + (2 * width), ours_6, width, yerr=ours_6_sem, label='Ours-6')

    ax.set_ylabel('Human Effort')
    ax.set_title('Human effort with and without assistance')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()