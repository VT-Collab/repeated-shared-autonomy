import pickle
import copy
import sys
import os
import numpy as np

def main(max_cnt):

    dataset = []
    folder = 'data/demos'
    lookahead = 5
    noiselevel = 0.05
    noisesamples = 5

    notepad_count = 0
    soupcan_count = 0
    tape_count = 0
    traj_cnt = 0
    glass_count = 0
    shelf_count = 0
    # max_cnt = 10
    # print(max_cnt)
    savename = 'data/task_3_cae_' + str(max_cnt) + '.pkl'
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        n_states = len(traj)
        add_data_flag = False
        if filename[0] == 'n':
            z = [1.0]
            if notepad_count < max_cnt:
                add_data_flag = True
            notepad_count += 1
        elif filename[0] == 'c':
            z = [0.0]
            if soupcan_count < max_cnt:
                add_data_flag = True
            soupcan_count += 1
        elif filename[0] == 't':
            z = [-1.0]
            if tape_count < max_cnt:
                add_data_flag = True
            tape_count += 1
        elif filename[0] == 'g':
            z = [-1.0]
            if glass_count < max_cnt:
                add_data_flag = True
            glass_count += 1
        elif filename[0] == 's':
            z = [-1.0]
            if shelf_count < max_cnt:
                add_data_flag = True
            shelf_count += 1

        if add_data_flag:
            for idx in range(n_states-lookahead):
                home_state = traj[idx][:7]
                position = np.asarray(traj[idx])[7:]
                nextposition = np.asarray(traj[idx + lookahead])[7:]
                for jdx in range(noisesamples):
                    action = nextposition - (position + np.random.normal(0, noiselevel, 7))
                    dataset.append((home_state + position.tolist(), position.tolist(), z, action.tolist()))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))



if __name__ == "__main__":
    for i in range(1, 6, 2):
        main(i)
