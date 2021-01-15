import pickle
import copy
import sys
import os
import numpy as np


def calculate_dist(state, current_pos):
    blue = state[0:2] #x2
    green = state[2:4] #x2
    start = state[6:8] #x1

    denom = np.sqrt((blue[0] - start[0])**2 + (blue[1] - start[1])**2)
    dist_1 = np.absolute((blue[0] - start[0]) * (start[1] - current_pos[1]) \
                - (start[0] - current_pos[0]) * (blue[1] - start[1]))/ denom
    
    denom = np.sqrt((green[0] - start[0])**2 + (green[1] - start[1])**2)
    dist_2 = np.absolute((green[0] - start[0]) * (start[1] - current_pos[1]) \
                - (start[0] - current_pos[0]) * (green[1] - start[1]))/ denom
    
    return min(dist_1, dist_2)

def main():

    dataset = []
    folder = 'data/demos'
    savename = 'data/dataset_with_fake_dist.pkl'
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
            position = np.random.random(2)
            pos_dist = calculate_dist(home_state, position)
            while pos_dist == 0:
                position =np.random.random(2)
                pos_dist = calculate_dist(home_state, position)

            dataset.append((home_state + position.tolist(),\
                             traj[idx], [z], action.tolist(), traj_type))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset)
    print("[*] I have this many subtrajectories: ", len(dataset))
    # avg = np.mean(actions, axis =0)
    # print("[*] This is the average action magnitude: ", np.amin(actions))



if __name__ == "__main__":
    main()
