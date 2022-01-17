from panda import Panda
from env import SimpleEnv
import numpy as np
import pybullet as p
import pybullet_data
import pickle
from scipy.interpolate import interp1d
import sys



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
    env_goals = sys.argv[1]
    goals = []
    radius = 0.6
    num_goals = int(env_goals)
    n_waypoints = 75
    rand_i = np.linspace(-0.25,0.25,num_goals+1)
    savename = "goals/goals" + str(num_goals) + ".pkl"


    # GENERATE A GIVEN NUMBER OF GOALS ON THE SURFACE OF A CYLINDER
    for i in range (num_goals):
        print(rand_i[i])
        g = get_cylinder(radius,rand_i[i])
        goals.append(g)
        

    pickle.dump(goals,open(savename, "wb"))
    goals = np.asarray(pickle.load(open(savename, "rb")))
    print("goals=", goals)
    env = SimpleEnv(env_goals)
    state = env.reset()
    # time.sleep(10)

    for goal in range (num_goals):
        print("GOAL NUMBER ", goal)
        file_name = "demos/Demos/" + str(goal+1) + ".pkl"
        filename = "demos/Noisy_Demos/" + str(goal+1) + "_0" + ".pkl"
        state = env.reset()
        state = state[-1]
        ee_pose = np.asarray(state['ee_position'])
        target = goals[goal]
        x_target = target[0]
        y_target = target[1]
        z_target = target[2]
        print(x_target, y_target, z_target)
        x_traj = np.linspace(ee_pose[0], x_target, n_waypoints)
        y_traj = np.linspace(ee_pose[1], y_target, n_waypoints)
        z_traj = np.linspace(ee_pose[2], z_target, n_waypoints)
        trajectory = np.column_stack((x_traj, y_traj, z_traj))
        traj_end = np.tile(target, (20,1))
        trajectory = np.row_stack((trajectory, traj_end))

        print(trajectory)

        pickle.dump(trajectory, open(file_name, "wb"))
        pickle.dump(trajectory, open(filename, "wb"))

    # folder = 'demos'
    # sapairs = []
    # for filename in os.listdir(folder):
    #     traj = pickle.load(open(folder + "/" + filename, "rb"))
    #     traj = np.asarray(traj)
    #     for idx in range(len(traj) - 1):
    #         s_base = traj[idx]
    #         sp = traj[idx + 1]
    #         for _ in range(10):
    #             s = np.copy(s_base) + np.random.normal(0, 0.1, 3)
    #             a = sp - s
    #             sapairs.append(s.tolist() + a.tolist())

    # # print(sapairs)
    # print(len(sapairs))
    # pickle.dump(sapairs, open("sa_pairs/sa_pairs_" + str(num_goals) + ".pkl", "wb"))








if __name__ == "__main__":
    main()