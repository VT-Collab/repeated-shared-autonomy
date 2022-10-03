import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    # Task 1
    dataset = []
    data_prefix = 'user/user'

    total_notepad_ours = []
    total_tape_ours = []
    total_tape_anca = []
    total_notepad_anca = []
    total_tape_none = []
    total_notepad_none = []

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
                total_notepad_none.append(data)

            if filename[0] == "t":
                total_tape_none.append(data)


    dataset = {}
    max_notepad = np.max([np.mean(total_notepad_ours), np.mean(total_notepad_anca)])
    max_tape = np.max([np.mean(total_tape_ours), np.mean(total_tape_anca)])
    ours = [np.mean(total_notepad_ours)/max_notepad, np.mean(total_tape_ours)/max_tape]
    anca = [np.mean(total_notepad_anca)/max_notepad, np.mean(total_tape_anca)/max_tape]
    # for item in total_tape_ours:
    #     print(item)

    # Task 2
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
    # for item in shelf_none:
    #     print(item)
    
    # Task 3
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
    for item in tape_none:
        print(item)

if __name__ == "__main__":
    main()