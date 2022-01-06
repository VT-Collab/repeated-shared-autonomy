from panda import Panda
from env import SimpleEnv
import numpy as np
import pybullet as p
import pybullet_data
import pickle
from scipy.interpolate import interp1d
import time
import math


def get_cylinder(radius,rand_i):
    # rand_i, rand_j = np.random.rand(2)      
    rand_j = np.random.uniform(0.0,0.5)
    theta = 2 * np.pi * rand_i  
    phi = np.pi/2 * rand_j
    x = radius*np.cos(theta) #* np.sin(phi)        
    y = radius*np.sin(theta) - 0.6 #* np.sin(phi) -0.6   
    z = radius*np.cos(phi)
    g = [x, y, z]
    # z = 0
    return g

def main():
    goals = []
    radius = 0.6
    num_goals = 2
    n_waypoints = 75
    rand_i = np.linspace(0.5,0.7,num_goals+1)
    savename = "goals" + str(num_goals) + ".pkl"


    # GENERATE A GIVEN NUMBER OF GOALS ON THE SURFACE OF A CYLINDER
    for i in range (num_goals):
        print(rand_i[i])
        g = get_cylinder(radius,rand_i[i])
        goals.append(g)
        

    pickle.dump(goals,open(savename, "wb"))
    goals = np.asarray(pickle.load(open(savename, "rb")))
    print("goals=", goals)
    env = SimpleEnv()
    state = env.reset()
    time.sleep(5)

    for goal in range (num_goals):
        print("GOAL NUMBER ", goal)
        for demo in range (5):
            file_name = "demos/" + str(goal+1) + "_" + str(demo+1) + ".pkl"
            state = env.reset()
            state = state[-1]
            ee_pose = np.asarray(state['ee_position'])
            target = goals[goal]
            x_target = np.random.normal(target[0], 0.005)
            y_target = np.random.normal(target[1], 0.005)
            z_target = np.random.normal(target[2], 0.005)
            target_noise = [x_target, y_target, z_target]
            print(x_target, y_target, z_target)
            x_traj = np.linspace(ee_pose[0], x_target, n_waypoints)
            y_traj = np.linspace(ee_pose[1], y_target, n_waypoints)
            z_traj = np.linspace(ee_pose[2], z_target, n_waypoints)
            trajectory = np.column_stack((x_traj, y_traj, z_traj))
            traj_end = np.tile(target_noise, (20,1))
            trajectory = np.row_stack((trajectory, traj_end))

            print(trajectory)

            pickle.dump(trajectory, open(file_name, "wb"))







if __name__ == "__main__":
    main()