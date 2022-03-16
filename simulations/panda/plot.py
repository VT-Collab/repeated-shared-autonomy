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
# plt.show()
# print(x1)
# print(traj2-traj1)

def main():

    num_goals = sys.argv[1]
    effort = np.zeros(int(num_goals))
    alpha = np.zeros(int(num_goals))
    error = np.zeros(int(num_goals))
    perc_error = np.zeros(int(num_goals))
    goals_plt = []
    start = [0.26021528, -0.60000002, 0.85696614]


    for goal in range (int(num_goals)):
        error_loc = 0
        error_per = 0
        goal_num = "goals/goals" + str(goal+1) + ".pkl"
        goals = pickle.load(open(goal_num, "rb"))
        for sim_num in range (5):
            filename_error = "effort/Error/" + str(goal+1) + "/" + str(sim_num+1) + ".pkl"
            error_arr = pickle.load(open(filename_error, "rb"))
            # error_loc = 0
            # error[goal] = np.linalg.norm(error_arr)
            for j in range (int(goal+1)):
                filename_effort = "effort/Effort/" + str(goal+1) + "/" + str(sim_num+1) + "_" + str(j+1) + ".pkl"
                filename_alpha = "effort/Alpha/" + str(goal+1) + "/" + str(sim_num+1) + "_" + str(j+1) + ".pkl"


                effort_arr = pickle.load(open(filename_effort, "rb"))
                alpha_arr = pickle.load(open(filename_alpha, "rb"))

                effort[goal] = effort[goal] + np.mean(effort_arr)
                alpha[goal] = alpha[goal] + np.mean(alpha_arr)
                error_loc = error_loc + np.linalg.norm(error_arr[j])


                start_error = np.asarray(goals[j]) - np.asarray(start)
                error_per = error_per + np.linalg.norm(error_arr[j])/np.linalg.norm(start_error)
                # print(error_arr[j])

        effort[goal] = effort[goal]/(5*(goal+1))
        alpha[goal] = alpha[goal]/(5*(goal+1))
        error[goal] = error_loc/(5*(goal+1))
        perc_error[goal] = 100*error_per/(5*(goal+1))
        goals_plt.append(goal+1)
    # print(effort)
    # print(alpha)

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))

    ax1.plot(goals_plt, effort)
    ax1.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(effort)*1.5), yticks=np.arange(0, max(effort)*1.5, max(effort)*0.15))
    ax1.set_xlabel("Number of Goals")
    ax1.set_ylabel("Average Effort")

    ax2.plot(goals_plt, alpha)
    ax2.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(alpha)*1.5), yticks=np.arange(0, max(alpha)*1.5, max(alpha)*0.15))
    ax2.set_xlabel("Number of Goals")
    ax2.set_ylabel("Average Alpha")

    ax3.plot(goals_plt, error)
    ax3.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(error)*1.5), yticks=np.arange(0, max(error)*1.5, max(error)*0.15))
    ax3.set_xlabel("Number of Goals")
    ax3.set_ylabel("Average Final Error")

    ax4.plot(goals_plt, perc_error)
    ax4.set(xlim=(0, int(num_goals)+1), xticks=np.arange(0, int(num_goals)+1), ylim=(0, max(perc_error)*1.5), yticks=np.arange(0, max(perc_error)*1.5, max(perc_error)*0.15))
    ax4.set_xlabel("Number of Goals")
    ax4.set_ylabel("Average Final Percent Error")
    plt.savefig('data.png')
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
