import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from sklearn.utils import shuffle
import sys
import os
import copy 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import rospy
import actionlib
import sys
import time
import numpy as np
import pygame
import pickle
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import copy
from collections import deque
from std_msgs.msg import Float64MultiArray, String

from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperFeedback, 
    CommandRobotiqGripperResult, 
    CommandRobotiqGripperAction, 
    CommandRobotiqGripperGoal
)

from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import (
    Robotiq2FingerGripperDriver as Robotiq
)

from controller_manager_msgs.srv import (
    SwitchController, 
    SwitchControllerRequest, 
    SwitchControllerResponse
)

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

from utils import go2home
from tasks import HOME


# plot3d traj
class KinematicsClient(object):

    def __init__(self):
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
      
    def joint2pose(self, q):
        state = self.kdl_kin.forward(q)
        xyz_lin = np.array(state[:,3][:3]).T
        xyz_lin = xyz_lin.tolist()
        R = state[:,:3][:3]
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = xyz = np.asarray(xyz_lin[-1]).tolist() + np.asarray(xyz_ang).tolist()
        return xyz

    def pose2joint(self, pose):
        return self.kdl_kin.inverse(pose, self.joint_states)

    def qdot2xdot(self, qdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        return J.dot(qdot)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

def main():
    rospy.init_node("scratchwork")
    
    kin_cli = KinematicsClient()
    
    tasklist = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
                      ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
                      ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"]]#,\
                      # ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
                      # ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
                      # ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                      # ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]
    
    tasks = ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2",\
                      "open1", "open2"] 

    folders = ["runs/alpha_0.8", "runs/alpha_nolimit"]
    # folders = ["runs/alpha_nolimit"]
    labels =["fixed blend", "alpha = [0,0.6]"]
    colors = ['tab:blue', 'tab:orange']
    for task in tasks:
        fig1, axs1 = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, sharex=True, sharey=True)
        fig2, axs2 = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, sharex=True, sharey=True)
        i = 0
        for folder in folders:
            for tasks, ax1, ax2 in zip(tasklist, axs1.ravel(), axs2.ravel()):
                mark = False
                for filename in os.listdir(folder):
                    if not filename[0] == "m":
                        continue
                    demo = pickle.load(open(folder + "/" + filename, "rb"))
                    model_name = "_".join(tasks)
                    if demo["model"] == model_name and demo["task"] == task:
                        traj = np.array([kin_cli.joint2pose(item[1]) for item in demo["data"]])
                        if not mark:
                            ax1.scatter(traj[0,0], traj[0,1], traj[0,2], 'rx')
                            ax2.scatter(traj[0,3], traj[0,4], traj[0,5], 'rx')
                            mark = True
                        ax1.plot(traj[:,0], traj[:,1], traj[:,2], color=colors[i], label=labels[i])
                        ax2.plot(traj[:,3], traj[:,4], traj[:,5], color=colors[i], label=labels[i])
                        if demo["run"] == 1:
                            individual_folder = "runs/individual_models"
                            fnames = ["model_" + task + "_task_" + task + "_" + str(demo_num) + ".pkl" for demo_num in range(1,6)]
                            for fname in fnames:
                                demo = pickle.load(open(individual_folder + "/" + fname, "rb"))
                                traj = np.array([kin_cli.joint2pose(item[1]) for item in demo["data"]])
                                # ax.view_init(elev=60, azim=60) 
                                ax1.plot(traj[:,0], traj[:,1], traj[:,2], color='tab:green', label="ideal")
                                ax2.plot(traj[:,3], traj[:,4], traj[:,5], color='tab:green', label="ideal")
            i += 1
        for i, (ax1, ax2) in enumerate(zip(axs1.ravel(), axs2.ravel())):
            ax1.title.set_text('Models trained: ' + str(i+1))
            ax2.title.set_text('Models trained: ' + str(i+1))

        
        handles, labels = ax1.get_legend_handles_labels()
        fig1.legend(handles, labels)
        fig2.legend(handles, labels)
        fig1.suptitle("XYZ trajectories for " + task + " with models trained from one task to 8 tasks")    
        fig2.suptitle("RPY trajectories for " + task + " with models trained from one task to 8 tasks")    
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

# for 

# plotting alphas

# tasklist = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
#                   ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
#                   ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
#                   ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]


# folder = "runs"
# mean_alphas = {}
# models = []
# for taskset in tasklist:
#     model = "_".join(taskset)
#     models.append(model)
#     alphas = {}
#     for filename in os.listdir(folder):
#         if not filename[0] == "m":
#             continue
#         traj = pickle.load(open(folder + "/" + filename, "rb"))
#         if not traj["model"] == model:
#             continue
#         task = traj["task"]
#         alpha = np.mean([item[-3] for item in traj["data"]])
#         if task in alphas:
#             alphas[task].append(alpha)
#         else:
#             alphas[task] = [float(alpha)]
#     # print(alphas)
#     means = []
#     for key in alphas:
#         means.append(np.mean(alphas[key]))
#     mean_alphas[model] = np.mean(means)

# sorted_alphas = [mean_alphas[model] for model in models]
# if len(sorted_alphas) <= 8:
#     print(sorted_alphas)
# else:
#     print(sorted_alphas[:8])
#     print(sorted_alphas[8:])


# def main():
#     rospy.init_node("scratchwork")
#     go2home()



# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass

# model_names = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
#                   ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
#                   ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
#                   ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]

# print(list(set(test_tasks).intersection(model_names[11])))


# data1 = pickle.load(open("demos/push2_1.pkl","rb"))

# # plotting deformations
# def deform(xi, start, length, tau):
#     length += 250
#     # length += np.random.choice(np.arange(50, 250))
#     xi1 = copy.deepcopy(np.asarray(xi))
#     A = np.zeros((length+2, length))
#     for idx in range(length):
#         A[idx, idx] = 1
#         A[idx+1,idx] = -2
#         A[idx+2,idx] = 1
#     R = np.linalg.inv(np.dot(A.T, A))
#     U = np.zeros(length)
#     gamma = np.zeros((length, len(tau)))
#     for idx in range(len(tau)):
#         U[0] = tau[idx]
#         gamma[:,idx] = np.dot(R, U)
#     end = min([start+length, xi1.shape[0]-1])
#     xi1[start:end+1,:] += gamma[0:end-start+1,:]
#     return xi1
# traj = np.array([item[0] for item in data1])
    
# fig = plt.figure()
# ax = fig.gca(projection='3d')    
# ax.plot(traj[:,6], traj[:,7], traj[:,8], 'r', linewidth=1.0)# label='parametric curve')
# tau = np.random.uniform([-0.00045]*6, [0.00045]*6)
# traj_def = deform(traj[:,6:], 0, len(traj), tau) 
# print(len(traj_def))
# ax.plot(traj_def[:,0], traj_def[:,1], traj_def[:,2], 'b', linewidth=2.0)#, label='parametric curve')
# plt.show()



# data1 = pickle.load(open("demos/push1_1.pkl","rb"))
# data2 = pickle.load(open("demos/push1_2.pkl","rb"))
# latent_dim = 5
#RNN
# gru = nn.GRU(
#     input_size = 18,
#     hidden_size = latent_dim,
#     num_layers = 1,
#     batch_first = True
#     )
# fcn = nn.Sequential(
#     nn.Linear(latent_dim, 5),
#     nn.Tanh(),
#     nn.Linear(5, 2)
# )
# traj = [item[0] + item[1] for item in data1]
#
# i1 = torch.Tensor(traj).unsqueeze(0)
# i1 = i1.unsqueeze(0)
# target = np.zeros(len(traj))
# print(target.shape)
# traj = [item[0] + item[1] for item in data2]
# i2 = torch.Tensor(traj)
# i2 = i2.unsqueeze(0)

# input = torch.cat((i1, i2), dim=0)
# print(input.shape)

# h, o = gru(input)
# y = fcn(h)
# print(y)
