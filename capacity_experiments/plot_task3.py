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

    cup_first = []
    cup_final = []
    cup_none = []
    tape_first = []
    tape_final = []
    tape_none = []
    notepad_first = []
    notepad_final = []
    notepad_none = []

    
    for usernumber in range(1,11):
        data_loc = data_prefix + str(usernumber) + '/runs'

        for filename in os.listdir(data_loc):
            data = 0.
            length = 0.
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

            if filename[0] == "r" and filename[4] == "n":
                notepad_first.append(data)

            if filename[0] == "r" and filename[4] == "t":
                tape_first.append(data)

            if filename[0] == "r" and filename[4] == "u":
                cup_first.append(data)

            if filename[0] == "f" and filename[6] == "n":
                notepad_final.append(data)

            if filename[0] == "f" and filename[6] == "t":
                tape_final.append(data)

            if filename[0] == "f" and filename[6] == "u":
                cup_final.append(data)

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
            if filename[0] == "n":
                notepad_none.append(data)

            if filename[0] == "u":
                cup_none.append(data)

            if filename[0] == "t":
                tape_none.append(data)

    first = {}
    first["notepad"] = np.mean(notepad_first)
    first["tape"] = np.mean(tape_first)
    first["cup"] = np.mean(cup_first)
    
    final = {}
    final["notepad"] = np.mean(notepad_final)
    final["tape"] = np.mean(tape_final)
    final["cup"] = np.mean(cup_final)

    dataset = [first, final]
    savename = 'data/consolidated_data/task3.pkl'
    pickle.dump(dataset, open(savename, "wb"))

    max_cup = np.max([np.mean(cup_none), np.mean(cup_first), np.mean(cup_final)])
    max_tape = np.max([np.mean(tape_none), np.mean(tape_first), np.mean(tape_final)])
    max_notepad = np.max([np.mean(notepad_none), np.mean(notepad_first), np.mean(notepad_final)])
    
    none = [np.mean(cup_none)/max_cup, np.mean(tape_none)/max_tape, np.mean(notepad_none)/max_notepad]
    first = [np.mean(cup_first)/max_cup, np.mean(tape_first)/max_tape, np.mean(notepad_first)/max_notepad]
    final = [np.mean(cup_final)/max_cup, np.mean(tape_final)/max_tape, np.mean(notepad_final)/max_notepad]

    none_sem = [np.std(cup_none)/(max_tape*np.sqrt(10)), np.std(tape_none)/(max_tape*np.sqrt(10)), np.std(notepad_none)/(max_notepad*np.sqrt(10))]
    first_sem = [np.std(cup_first)/(max_tape*np.sqrt(10)), np.std(tape_first)/(max_tape*np.sqrt(10)), np.std(notepad_first)/(max_notepad*np.sqrt(10))]
    final_sem = [np.std(cup_final)/(max_tape*np.sqrt(10)), np.std(tape_final)/(max_tape*np.sqrt(10)), np.std(notepad_final)/(max_notepad*np.sqrt(10))]
    
    labels = ['Cup', 'Tape', 'Notepad']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, none, width, yerr=none_sem, label='No Assist')
    rects2 = ax.bar(x + width, first, width, yerr=first_sem, label='First')
    rects3 = ax.bar(x + (2 * width), final, width, yerr=final_sem, label='Final')

    ax.set_ylabel('Human Effort')
    ax.set_title('Human effort with and without assistance')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()