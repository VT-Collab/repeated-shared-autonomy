import pickle
import copy
import sys
import os
import numpy as np


def main():

    dataset = []
    folder = 'data/demos'
    savename = 'data/dataset.pkl'
    lookahead = 5
    noiselevel = 0.05
    noisesamples = 5

    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        n_states = len(traj)
        if filename[0] == 'r':
            z = [1.0]
        else:
            z = [-1.0]
        home_state = traj[0]
        for idx in range(n_states-lookahead):
            obj_features = np.asarray(traj[idx])[0:4]
            position = np.asarray(traj[idx])[4:]
            nextposition = np.asarray(traj[idx + lookahead])[4:]
            for jdx in range(noisesamples):
                action = nextposition - (position + np.random.normal(0, noiselevel, 7))
                dataset.append((home_state + position.tolist(), obj_features.tolist() + position.tolist(), z, action.tolist()))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset)
    print("[*] I have this many subtrajectories: ", len(dataset))



if __name__ == "__main__":
    main()
