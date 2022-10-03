import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Storing some important states
HOME = np.asarray([-0.000453, -0.7853, 0.000176, -2.355998, -0.000627, 1.572099, 0.7859])
SOUPCAN = np.asarray([-0.305573, 0.70963, -0.183403, -1.500632, 0.13223, 2.214831, 0.270835])
NOTEPAD = np.asarray([0.344895, 0.796234, 0.227432, -1.514207, -0.195541, 2.286966, 1.421964])
# CUP = [np.asarray([-0.169027, 0.119563, -0.662496, -2.628833, 0.120918, 2.729636, -0.146264])]
TAPE = np.asarray([-0.016421, 0.210632, 0.031353, -2.594983, 0.020064, 2.816127, 0.877912])
CUP = np.asarray([-0.013822, 0.852029, 0.058359, -1.310726, -0.054624, 2.162042, 0.835634])

def joint2pose(q):
    def RotX(q):
        return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
    def RotZ(q):
        return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def TransX(q, x, y, z):
        return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
    def TransZ(q, x, y, z):
        return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    H1 = TransZ(q[0], 0, 0, 0.333)
    H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
    H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
    H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
    H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
    H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
    H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
    H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
    H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
    return H[:,3][:3]

SOUPCAN_C = joint2pose(SOUPCAN)[:3]
NOTEPAD_C = joint2pose(NOTEPAD)[:3]
TAPE_C = joint2pose(TAPE)[:3]

def main():

    # print(TAPE_C)
    dataset = []
    data_loc = 'data/test_runs/task1/'

    dist_can_ours_1 = []
    dist_can_ours_3 = []
    dist_can_ours_5 = []
    dist_note_ours_1 = []
    dist_note_ours_3 = []
    dist_note_ours_5 = []
    dist_tape_ours_1 = []
    dist_tape_ours_3 = []
    dist_tape_ours_5 = []

    dist_can_bc_1 = []
    dist_can_bc_3 = []
    dist_can_bc_5 = []
    dist_note_bc_1 = []
    dist_note_bc_3 = []
    dist_note_bc_5 = []
    dist_tape_bc_1 = []
    dist_tape_bc_3 = []
    dist_tape_bc_5 = []

    # data = []
    for filename in os.listdir(data_loc):
        # if filename[0] == "c" or filename[0] == "n" or filename[0] == "t":
        traj = pickle.load(open(data_loc + "/" + filename, "rb"))
        point = traj[-1]
        state = np.asarray(point[1])
        state = joint2pose(state)[:3]

        if filename[0] == "c" and filename[2] == "o" and filename[7] == "1":
            dist_can_ours_1.append(np.linalg.norm(state - SOUPCAN_C))
        elif filename[0] == "c" and filename[2] == "o" and filename[7] == "3":
            dist_can_ours_3.append(np.linalg.norm(state - SOUPCAN_C))
        elif filename[0] == "c" and filename[2] == "o" and filename[7] == "5":
            dist_can_ours_5.append(np.linalg.norm(state - SOUPCAN_C))
        
        elif filename[0] == "n" and filename[2] == "o" and filename[7] == "1":
            dist_note_ours_1.append(np.linalg.norm(state - NOTEPAD_C))
        elif filename[0] == "n" and filename[2] == "o" and filename[7] == "3":
            dist_note_ours_3.append(np.linalg.norm(state - NOTEPAD_C))
        elif filename[0] == "n" and filename[2] == "o" and filename[7] == "5":
            dist_note_ours_5.append(np.linalg.norm(state - NOTEPAD_C))

        elif filename[0] == "t" and filename[2] == "o" and filename[7] == "1":
            dist_tape_ours_1.append(np.linalg.norm(state - TAPE_C))
        elif filename[0] == "t" and filename[2] == "o" and filename[7] == "3":
            dist_tape_ours_3.append(np.linalg.norm(state - TAPE_C))
        elif filename[0] == "t" and filename[2] == "o" and filename[7] == "5":
            dist_tape_ours_5.append(np.linalg.norm(state - TAPE_C))

        elif filename[0] == "c" and filename[2] == "b" and filename[7] == "1":
            dist_can_bc_1.append(np.linalg.norm(state - SOUPCAN_C))
        elif filename[0] == "c" and filename[2] == "b" and filename[7] == "3":
            dist_can_bc_3.append(np.linalg.norm(state - SOUPCAN_C))
        elif filename[0] == "c" and filename[2] == "b" and filename[7] == "5":
            dist_can_bc_5.append(np.linalg.norm(state - SOUPCAN_C))
        
        elif filename[0] == "n" and filename[2] == "b" and filename[7] == "1":
            dist_note_bc_1.append(np.linalg.norm(state - NOTEPAD_C))
        elif filename[0] == "n" and filename[2] == "b" and filename[7] == "3":
            dist_note_bc_3.append(np.linalg.norm(state - NOTEPAD_C))
        elif filename[0] == "n" and filename[2] == "b" and filename[7] == "5":
            dist_note_bc_5.append(np.linalg.norm(state - NOTEPAD_C))

        elif filename[0] == "t" and filename[2] == "b" and filename[7] == "1":
            dist_tape_bc_1.append(np.linalg.norm(state - TAPE_C))
        elif filename[0] == "t" and filename[2] == "b" and filename[7] == "3":
            dist_tape_bc_3.append(np.linalg.norm(state - TAPE_C))
        elif filename[0] == "t" and filename[2] == "b" and filename[7] == "5":
            dist_tape_bc_5.append(np.linalg.norm(state - TAPE_C))


    # print(len(dist_note_ours_5))
    ours = {}
    ours["c"] = dist_can_ours_5
    ours["n"] = dist_note_ours_5
    ours["t"] = dist_tape_ours_5

    bc = {}
    bc["c"] = dist_can_bc_5
    bc["n"] = dist_note_bc_5
    bc["t"] = dist_tape_bc_5

    dataset = {}
    dataset["5"] = [ours, bc]

    ours = {}
    ours["c"] = dist_can_ours_3
    ours["n"] = dist_note_ours_3
    ours["t"] = dist_tape_ours_3

    bc = {}
    bc["c"] = dist_can_bc_3
    bc["n"] = dist_note_bc_3
    bc["t"] = dist_tape_bc_3

    # print(bc)
    dataset["3"] = [ours, bc]
    savename = "data/consolidated_data/task1.pkl"
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset["3"])
    # fig, axs = plt.subplots(1, 3, figsize=(15, 9), sharex=True)
    # trials_can = [sum(dist_cup_bc_3)/len(dist_cup_bc_3), sum(dist_cup_ours_3)/len(dist_cup_ours_3), sum(dist_cup_gold_3)/len(dist_cup_gold_3)]
    # x = np.arange(len(trials_can))
    # labels = ['behavior cloning', 'ours', 'gold']
    # axs[0].bar(x, trials_can)
    # axs[0].set_ylabel('Distance')
    # axs[0].set_title('Distance from can(known goal) after 3 demos')
    # axs[0].set_xticks(x)
    # axs[0].set_xticklabels(labels)
    # trials_notepad = [sum(dist_note_bc_3)/len(dist_note_bc_3), sum(dist_note_ours_3)/len(dist_note_ours_3), sum(dist_note_gold_3)/len(dist_note_gold_3)]
    # axs[1].bar(x, trials_notepad)
    # axs[1].set_ylabel('Distance')
    # axs[1].set_title('Distance from notepad (known goal) after 3 demos')
    # axs[1].set_xticks(x)
    # axs[1].set_xticklabels(labels)
    # trials_tape = [sum(dist_tape_bc_3)/len(dist_tape_bc_3), sum(dist_tape_ours_3)/len(dist_tape_ours_3), sum(dist_tape_gold_3)/len(dist_tape_gold_3)]
    # axs[2].bar(x, trials_tape)
    # axs[2].set_ylabel('Distance')
    # axs[2].set_title('Distance from tape (known goal) after 3 demos')
    # axs[2].set_xticks(x)
    # axs[2].set_xticklabels(labels)

    # fig.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
