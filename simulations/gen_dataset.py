from panda import Panda
from env import SimpleEnv
import numpy as np
import pybullet as p
import pybullet_data
import pickle
from scipy.interpolate import interp1d

def main():
    goals = []
    radius = 0.6
    num_goals = 15
    n_waypoints = 75
    kind = "linear"
    
    # GENERATE A GIVEN NUMBER OF GOALS ON THE SURFACE OF A SPHERE
    # for _ in range (num_goals):
    #     rand_i, rand_j = np.random.rand(2)                 
    #     theta = 2 * np.pi * rand_i  
    #     phi = np.arccos(abs(2 * rand_j - 1)) 
    #     x = radius*np.cos(theta) * np.sin(phi)        
    #     y = radius*np.sin(theta) * np.sin(phi)        
    #     z = radius*np.cos(phi)
    #     goals.append([x, y, z])

    # pickle.dump(goals,open("goals.pkl", "wb"))
    goals = np.asarray(pickle.load(open("goals.pkl", "rb")))
    # print("goals=", goals)
    env = SimpleEnv()
    state = env.reset()

    for goal in range (num_goals):
        print("GOAL NUMBER ", goal)
        for demo in range (5):
            file_name = "Demos/" + str(goal+1) + "_" + str(demo+1) + "pkl"
            state = env.reset()
            state = state[-1]
            ee_pose = np.asarray(state['ee_position'])
            target = goals[goal]
            x_target = np.random.normal(target[0], 0.025)
            y_target = np.random.normal(target[1], 0.025)
            z_target = np.random.normal(target[2], 0.025)
            print(x_target, y_target, z_target)
            x_traj = np.linspace(ee_pose[0], x_target, n_waypoints)
            y_traj = np.linspace(ee_pose[1], y_target, n_waypoints)
            z_traj = np.linspace(ee_pose[2], z_target, n_waypoints)
            trajectory = np.array([x_traj, y_traj, z_traj])
            # print(trajectory)

            pickle.dump(trajectory, open(file_name, "wb"))







if __name__ == "__main__":
    main()