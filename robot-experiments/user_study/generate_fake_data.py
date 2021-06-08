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


def main():

    user = sys.argv[1]
    demos = int(sys.argv[2])

    dataset = []
    folder = 'user/user' + str(user) + '/demos'
    lookahead = 0
    noiselevel = 0.05
    # noisesamples = 3

    true_cnt = 0
    false_cnt = 0

    notepad_count = 0
    soupcan_count = 0
    tape_count = 0
    cup_count = 0
    traj_cnt = 0
    shelf_count = 0

    savename = 'data/user' + str(user) + '_class_' + str(demos) + '.pkl'
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        n_states = len(traj)
        add_data_flag = False
        if filename[0] == 'n':
            z = [1.0]
            if notepad_count < 5:
                add_data_flag = True
                print("added notepad")
            notepad_count += 1
        elif filename[0] == 't':
            z = [-1.0]
            if tape_count < 5:
                add_data_flag = True
                print("added tape")
            tape_count += 1
        elif filename[0] == 's':
            z = [0.0]
            if shelf_count < demos:
                add_data_flag = True
                print("added shelf")
            shelf_count += 1

        elif filename[0] == 'u':
            z = [0.0]
            if cup_count < demos:
                add_data_flag = True
                print("added cup")
            cup_count += 1

        # elif filename[0] == 'g':
        #     z = [0.0]
        #     if soupcan_count < max_cnt:
        #         add_data_flag = True
        #     soupcan_count += 1



        if add_data_flag:
            for idx in range(n_states-lookahead):
                home_state = joint2pose(traj[idx][:7]).tolist()
                position = joint2pose(np.asarray(traj[idx])[7:])
                nextposition = joint2pose(np.asarray(traj[idx + lookahead])[7:])
                action = nextposition - (position + np.random.normal(0, noiselevel, 3))
                traj_type = 0
                dataset.append((home_state + position.tolist(), position.tolist(), z, action.tolist(), traj_type))
                true_cnt += 1

            snippets = np.array_split(traj, 2)
            # snippets = [traj]
            if demos == 0:
                deformed_samples = 2
            else:
                deformed_samples = 2
            for snip in snippets:
                tau = np.random.uniform([-0.07]*7, [0.05]*7)
                deform_len = len(snip)
                # print(deform_len)
                start = 0
                for i in range(deformed_samples):
                    snip_deformed = deform(snip[:,7:], 0, deform_len, tau)
                    snip[:,7:] = snip_deformed
                    # fake data
                    n_states = len(snip)
                    for idx in range(start, deform_len):
                        home_state = joint2pose(snip[idx][:7]).tolist()
                        position = joint2pose(np.asarray(snip[idx])[7:]) 
                        #position = position + np.random.normal(0, noiselevel, 7)
                        action = 0
                        traj_type = 1
                        dataset.append((home_state + position.tolist(), position.tolist(), z, action, traj_type))
                        false_cnt += 1
                    # print(dataset[-1])
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[-1])
    print("[*] I have this many subtrajectories: ", len(dataset))
    print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))
    # print(traj_cnt)


if __name__ == "__main__":
    main()
