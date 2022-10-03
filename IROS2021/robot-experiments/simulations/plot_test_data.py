import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():

    dataset = []
    data_loc = 'data/test_runs/data/30_demos/'

    # confidence = []
    # time = []
    # human_only = []
    # human_robot = []
    # robot_only = []

    time_ours_t = []
    time_bc_t = []
    time_gold_t = []
    time_dropout_t = []

    time_ours_u = []
    time_bc_u = []
    time_gold_u = []
    time_dropout_u = []

    time_ours_c = []
    time_bc_c = []
    time_gold_c = []
    time_dropout_c = []

    time_ours_n = []
    time_bc_n = []
    time_gold_n = []
    time_dropout_n = []

    # flag = False
    for folder in os.listdir(data_loc):
        for filename in os.listdir(data_loc + folder):
            traj = pickle.load(open(data_loc + folder + "/" + filename, "rb"))
            point = traj[-1]

            goal = "t"
            if folder == "ours" and filename[0] == goal:
                time_ours_t.append(point[0])
            elif folder == "bc" and filename[0] == goal:
                time_bc_t.append(point[0])
            elif folder == "gold" and filename[0] == goal:
                time_gold_t.append(point[0])
            elif folder == "dropout" and filename[0] == goal:
                time_dropout_t.append(point[0])

            goal = "u"
            if folder == "ours" and filename[0] == goal:
                time_ours_u.append(point[0])
            elif folder == "bc" and filename[0] == goal:
                time_bc_u.append(point[0])
            elif folder == "gold" and filename[0] == goal:
                time_gold_u.append(point[0])
            elif folder == "dropout" and filename[0] == goal:
                time_dropout_u.append(point[0])

            goal = "n"
            if folder == "ours" and filename[0] == goal:
                time_ours_n.append(point[0])
            elif folder == "bc" and filename[0] == goal:
                time_bc_n.append(point[0])
            elif folder == "gold" and filename[0] == goal:
                time_gold_n.append(point[0])
            elif folder == "dropout" and filename[0] == goal:
                time_dropout_n.append(point[0])

            goal = "c"
            if folder == "ours" and filename[0] == goal:
                time_ours_c.append(point[0])
            elif folder == "bc" and filename[0] == goal:
                time_bc_c.append(point[0])
            elif folder == "gold" and filename[0] == goal:
                time_gold_c.append(point[0])
            elif folder == "dropout" and filename[0] == goal:
                time_dropout_c.append(point[0])


        
        # if filename[0] == 'n' and not flag:
        #     traj = pickle.load(open(folder + "/" + filename, "rb"))
        #     for point in traj:
        #        time.append(point[0])
        #        confidence.append(point[-1])
        #        human_input = np.asarray(point[2])
        #        robot_input = np.asarray(point[3])
        #        human_only.append(np.sum(np.absolute(human_input)))
        #        robot_only.append(np.sum(np.absolute(robot_input) * point[-1]))
        #        human_robot.append(np.sum(np.absolute(human_input) * (1 - point[-1])))
        #     flag =True
    # folder = 'data/test_runs/30_demos'

    # confidence2 = []
    # time2 = []
    # human_only2 = []
    # human_robot2 = []
    # robot_only2 = []
    # flag = False
    # for filename in os.listdir(folder):
    #     if filename[0] == 'n' and not flag:
    #         traj = pickle.load(open(folder + "/" + filename, "rb"))
    #         for point in traj:
    #            time2.append(point[0])
    #            confidence2.append(point[-1])
    #            human_input2 = np.asarray(point[2])
    #            robot_input2 = np.asarray(point[3])
    #            human_only2.append(np.sum(np.absolute(human_input2)))
    #            robot_only2.append(np.sum(np.absolute(robot_input2) * point[-1]))
    #            human_robot2.append(np.sum(np.absolute(human_input2) * (1 - point[-1])))
    #         flag =True
    # print(confidence)
    # print("[*] I have this many subtrajectories: ", len(traj))

    fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharex = True)

    mean_time = [sum(time_gold_c)/len(time_gold_c), sum(time_dropout_c)/len(time_dropout_c),\
                sum(time_bc_c)/len(time_bc_c), sum(time_ours_c)/len(time_ours_c)]
    x = np.arange(len(mean_time))
    width = 0.35

    labels = ['gold standard', 'dropout', 'behavior cloning', 'ours']
    axs[0][0].bar(x, mean_time)
    axs[0][0].set_ylabel('Time(s)')
    axs[0][0].set_title('Time to reach soup can (known goal)')
    axs[0][0].set_xticks(x)
    axs[0][0].set_xticklabels(labels)

    mean_time = [sum(time_gold_n)/len(time_gold_n), sum(time_dropout_n)/len(time_dropout_n),\
                sum(time_bc_n)/len(time_bc_n), sum(time_ours_n)/len(time_ours_n)]

    axs[0][1].bar(x, mean_time)
    axs[0][1].set_ylabel('Time(s)')
    axs[0][1].set_title('Time to reach notepad (known goal)')
    axs[0][1].set_xticks(x)
    axs[0][1].set_xticklabels(labels)

    mean_time = [sum(time_gold_t)/len(time_gold_t), sum(time_dropout_t)/len(time_dropout_t),\
                sum(time_bc_t)/len(time_bc_t), sum(time_ours_t)/len(time_ours_t)]

    axs[1][0].bar(x, mean_time)
    axs[1][0].set_ylabel('Time(s)')
    axs[1][0].set_title('Time to reach measuring tape (known goal)')
    axs[1][0].set_xticks(x)
    axs[1][0].set_xticklabels(labels)

    mean_time = [0, sum(time_dropout_u)/len(time_dropout_u),\
                sum(time_bc_u)/len(time_bc_u), sum(time_ours_u)/len(time_ours_u)]
    axs[1][1].bar(x, mean_time)
    axs[1][1].set_ylabel('Time(s)')
    axs[1][1].set_title('Time to reach coffee cup (unknown goal)')
    axs[1][1].set_xticks(x)
    axs[1][1].set_xticklabels(labels)

    # axs[1].plot(time2, human_robot2)
    # axs[1].plot(time2, robot_only2)

    # axs[0].plot(time, confidence)
    # axs[0].set_title("Confidence vs. time")
    # axs[0].set_xlabel('time')
    # axs[0].set_ylabel('confidence')

    # l2 = axs[1].plot(time, human_only, label="human only")
    # l4 = axs[1].plot(time, human_robot, label="human with robot")
    # axs[1].set_title("human effort")
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('action magnitude')
    # axs[1].legend(loc='upper right')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
