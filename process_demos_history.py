import pickle
import copy
import sys
import os
import numpy as np

def main():

    dataset = []
    folder = 'data/demos'
    savename = 'data/dataset_traj.pkl'
    scale = 10.0
    for filename in os.listdir(folder):
        traj_raw = pickle.load(open(folder + "/" + filename, "rb"))
        
        # Exclude non-unique traj points
        traj_positions = np.asarray(traj_raw)[:,6:8]
        _, indices = np.unique(traj_positions,return_index=True, axis=0)
        indices  = np.sort(indices)
        traj = [traj_raw[i] for i in indices]
        # break
        if filename[0] == 'b':
            z = 1
        else:
            z = 0
        goal_loc = traj[0][:6]
        n_states = len(traj)
        for idx in range(n_states-3):
            pos_1 = np.asarray(traj[idx])[6:8]

            next_idx = idx + 1
            pos_2 = np.asarray(traj[next_idx])[6:8]
            next_idx += 1
            pos_3 = np.asarray(traj[next_idx])[6:8]
            positions = pos_1.tolist() + pos_2.tolist() + pos_3.tolist()
            # dist = np.min(np.linalg.norm(position-pos_2), np.linalg.norm(pos_2-pos_3)) 
            action = scale * (np.asarray(traj[next_idx + 1])[6:8] - pos_3)
            # print(action)
            traj_type = 0
            dataset.append((goal_loc + positions,\
                             traj[next_idx], [z], action.tolist(), traj_type))
            # Fake data
            traj_type = 1
            
            # Generate fake action
            action = np.random.uniform(low=-0.5,high=0.5,size=2)
            # print(action)
            pos_2 = pos_1 + (action/scale)
            # action = np.random.uniform(low=-1.0,high=1.0,size=2)
            pos_3 = pos_3 + (action/scale)
            positions = pos_1.tolist() + pos_2.tolist() + pos_3.tolist()

            dataset.append((goal_loc + positions,\
                             traj[next_idx], [z], action.tolist(), traj_type))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))



if __name__ == "__main__":
    main()
