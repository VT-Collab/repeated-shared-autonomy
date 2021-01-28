import pickle
import copy
import sys
import os
import numpy as np

def main():

    dataset = []
    folder = 'data/demos'
    savename = 'data/dataset_low_sigma.pkl'
    scale = 10.0
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        n_states = len(traj)
        if filename[0] == 'b':
            z = 1
        else:
            z = 0
        home_state = traj[0]
        for idx in range(n_states-1):
            position = np.asarray(traj[idx])[6:8]
            action = scale * (np.asarray(traj[idx + 1])[6:8] - position)
            traj_type = 0
            dataset.append((home_state + position.tolist(),\
                             traj[idx], [z], action.tolist(), traj_type))
            # Fake data
            traj_type = 1
            covariance = [[0.05, 0], [0, 0.05]]
            position = np.random.multivariate_normal(position, covariance)

            dataset.append((home_state + position.tolist(),\
                             traj[idx], [z], action.tolist(), traj_type))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset)
    print("[*] I have this many subtrajectories: ", len(dataset))



if __name__ == "__main__":
    main()
