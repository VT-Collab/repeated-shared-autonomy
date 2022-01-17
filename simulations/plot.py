import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import sys

# goals = pickle.load(open("goals/goals3.pkl", "rb"))
# traj1 = np.asarray(pickle.load(open("demos/Noisy_Demos/1_1.pkl", "rb")))
# traj2 = np.asarray(pickle.load(open("demos/Noisy_Demos/2_1.pkl", "rb")))
# traj3 = np.asarray(pickle.load(open("demos/Noisy_Demos/3_1.pkl", "rb")))


# # print(goals)
# x1 = traj1[:,0]
# y1 = traj1[:,1]
# z1 = traj1[:,2]

# x2 = traj2[:,0]
# y2 = traj2[:,1]
# z2 = traj2[:,2]

# x3 = traj3[:,0]
# y3 = traj3[:,1]
# z3 = traj3[:,2]

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x1, y1, z1)
# ax.plot3D(x2, y2, z2)
# ax.plot3D(x3, y3, z3)
plt.show()
# print(x1)
# print(traj2-traj1)

def main():
    
    num_goals = sys.argv[1]
    effort = np.zeros(int(num_goals))
    alpha = np.zeros(int(num_goals))
    error = np.zeros(int(num_goals))
    goals = []
    
    for goal in range (int(num_goals)):
        filename_error = "effort/Error/" + str(goal+1) + ".pkl"
        error_arr = pickle.load(open(filename_error, "rb"))
        error_loc = 0
        # error[goal] = np.linalg.norm(error_arr)
        for j in range (int(goal+1)):
            filename_effort = "effort/Effort/" + str(goal+1) + "_" + str(j+1) + ".pkl"
            filename_alpha = "effort/Alpha/" + str(goal+1) + "_" + str(j+1) + ".pkl"

            effort_arr = pickle.load(open(filename_effort, "rb"))
            alpha_arr = pickle.load(open(filename_alpha, "rb"))

            effort[goal] = effort[goal] + np.mean(effort_arr) 
            alpha[goal] = alpha[goal] + np.mean(alpha_arr)
            error_loc = error_loc + np.linalg.norm(error_arr[j])
            print(error_arr[j])

        effort[goal] = effort[goal]/(goal+1)
        alpha[goal] = alpha[goal]/(goal+1)
        error[goal] = error_loc/(goal+1)
        goals.append(goal+1)
    print(effort)
    print(alpha)

    fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5,10))
    
    ax1.plot(goals, effort)
    ax1.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(effort)*1.5), yticks=np.arange(0, max(effort)*1.5, max(effort)*0.15))
    ax1.set_xlabel("Number of Goals")
    ax1.set_ylabel("Average Effort")

    ax2.plot(goals, alpha)
    ax2.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(alpha)*1.5), yticks=np.arange(0, max(alpha)*1.5, max(alpha)*0.15))
    ax2.set_xlabel("Number of Goals")
    ax2.set_ylabel("Average Alpha")
   
    ax3.plot(goals, error)
    ax3.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(error)*1.5), yticks=np.arange(0, max(error)*1.5, max(error)*0.15))
    ax3.set_xlabel("Number of Goals")
    ax3.set_ylabel("Average Final Error")
    plt.show()

    # fig4= plt.figure()
    # ax4 = plt.axes(projection='3d')
    # ax4.plot3D(alpha, error, goals)
    # # ax = fig.add_axes()
    # # ax.bar(goals,effort)
    # # plt.plot(alpha, error)
    # ax4.set(xlim=(0, max(alpha)*1.5), xticks=np.arange(0, max(alpha)*1.5, max(alpha)*0.15), 
    #         ylim=(0, max(error)*1.5), yticks=np.arange(0, max(error)*1.5, max(error)*0.15),
    #         zlim=(0, int(num_goals)+1), zticks=np.arange(0, int(num_goals)+1))
    # # plt.xlabel("Average Alpha")
    # # plt.ylabel("Average Final Error")
    # ax4.set_xlabel('Average Alpha')
    # ax4.set_ylabel('Average Final Error')
    # ax4.set_zlabel('Numebr of Goals')
    # plt.show()

        
            
            

if __name__ == "__main__":
    main()


