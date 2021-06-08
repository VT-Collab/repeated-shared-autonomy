import pickle
import copy
import sys
import os
import numpy as np

def deform(xi, start, length, tau):
    xi1 = copy.deepcopy(np.asarray(xi))
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(A.T @ A)
    U = np.zeros(length)
    gamma = np.zeros((length, 7))
    for idx in range(7):
        U[0] = tau[idx]
        gamma[:,idx] = R @ U
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end,:] += gamma[0:end-start,:]
    return xi1

def main(max_cnt):

    dataset = []
    folder = 'data/demos'
    lookahead = 0
    noiselevel = 0.05
    # noisesamples = 3

    true_cnt = 0
    false_cnt = 0

    notepad_count = 0
    soupcan_count = 0
    tape_count = 0
    glass_count = 0
    traj_cnt = 0
    shelf_count = 0
    print(max_cnt)
    savename = 'data/task_3_class_' + str(max_cnt) + '.pkl'
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
            z = [0.5]
            if soupcan_count < max_cnt:
                add_data_flag = True
            soupcan_count += 1
        elif filename[0] == 't':
            z = [-1.0]
            if tape_count < max_cnt:
                add_data_flag = True
                print("added tape")
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
                print("added shelf")
            shelf_count += 1


        if add_data_flag:
            traj_cnt += 1
            for idx in range(n_states):
                home_state = traj[idx][:7]
                position = np.asarray(traj[idx])[7:]
                nextposition = np.asarray(traj[idx + lookahead])[7:]
                # for jdx in range(noisesamples):
                action = nextposition - (position + np.random.normal(0, noiselevel, 7))
                traj_type = 0
                dataset.append((home_state + position.tolist(), position.tolist(), z, action.tolist(), traj_type))
                true_cnt += 1

            snippets = np.array_split(traj, 2)
            # snippets = [traj]
            deformed_samples = 2
            for snip in snippets:
                tau = np.random.uniform([-0.07]*7, [0.07]*7)
                deform_len = len(snip)
                # print(deform_len)
                start = 0
                for i in range(deformed_samples):
                    snip_deformed = deform(snip[:,7:], 0, deform_len, tau)
                    # print(np.linalg.norm(snip[:, 7:] - snip_deformed))
                    snip[:,7:] = snip_deformed
                    # fake data
                    n_states = len(snip)
                    for idx in range(start, deform_len):
                        home_state = snip[idx][:7].tolist()
                        position = np.asarray(snip[idx])[7:] 
                        # nextposition = np.asarray(snip[idx + lookahead])[7:]
                        # for jdx in range(noisesamples):
                        position = position #+ np.random.normal(0, noiselevel, 7)
                        action = nextposition - (position + np.random.normal(0, noiselevel, 7))
                        traj_type = 1
                        dataset.append((home_state + position.tolist(), position.tolist(), z, action.tolist(), traj_type))
                        false_cnt += 1
                    # print(dataset[-1])
    pickle.dump(dataset, open(savename, "wb"))
    # print(dataset[1])
    print("[*] I have this many subtrajectories: ", len(dataset))
    print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))
    # print(traj_cnt)


if __name__ == "__main__":
    for i in range(5, 6, 2):
        main(i)
