# Standard imports
import rospy, actionlib, copy
import sys, time, pygame, pickle
import numpy as np
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from std_msgs.msg import Float64MultiArray, String
from train_classifier_old import Net
# ROS required imports
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

# Imports from current directory
from utils import TrajectoryClient, JoystickControl, get_human_action, compute_reward
from tasks import HOME, TASKSET 


class Model(object):

    def __init__(self, classifier_name):
        self.class_net = Net()
        # self.cae_net = CAE()
        
        model_dict = torch.load(classifier_name, map_location='cpu')
        self.class_net.load_state_dict(model_dict)
        
        # model_dict = torch.load(cae_name, map_location='cpu')
        # self.cae_net.load_state_dict(model_dict)

        self.class_net.eval
        # self.cae_net.eval

    def classify(self, c):
        labels = self.class_net.classifier(torch.FloatTensor(c))
        confidence = F.softmax(labels, dim=0)
        return confidence.data[0].numpy()

    # def encoder(self, c):
    #     z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
    #     return z_mean_tensor.tolist()

    # def decoder(self, z, s):
    #     z_tensor = torch.FloatTensor(z + s)
    #     a_predicted = self.cae_net.decoder(z_tensor)
    #     return a_predicted.data.numpy()


def run_test(model_name, test_task):
    rospy.init_node("test_method_old")

    mover = TrajectoryClient()
    joystick = JoystickControl()

    start_time = time.time()
    rate = rospy.Rate(1000)

    if np.linalg.norm(np.array(HOME) - np.array(mover.joint_states)) > 0.01:
        mover.switch_controller(mode='position')
        mover.send_joint(HOME, 5.0)
        mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')

    # cae_model = 'models/' + 'cae_' + str(model_name)
    class_model = 'models/' + 'class_' + str(model_name) + "_old"
    model = Model(class_model)#, cae_model)

    step_time  = 0.1
    scaling_trans = 0.1
    scaling_rot = 0.2
    start_q = mover.joint_states
    start_pos = mover.joint2pose()
    goals = TASKSET[test_task]
    goal_idx = 0
    traj = []
    alphas = []
    while not rospy.is_shutdown():

        q = np.asarray(mover.joint_states).tolist()
        s = mover.joint2pose()
        axes, start, mode, stop = joystick.getInput()
        if stop or len(traj) >= 250:
            # pickle.dump(demonstration, open(filename, "wb"))
            # # print(demonstration)
            # print("[*] Done!")
            # print("[*] I recorded this many datapoints: ", len(demonstration))
            mover.switch_controller(mode='position')
            mover.send_joint(q, 1.0)
            return float(np.mean(alphas))

        # xdot_h = np.zeros(6)
        # if mode:
        #     xdot_h[3:] = scaling_trans * np.asarray(axes)
        # else:
        #     xdot_h[:3] = scaling_rot * np.asarray(axes)
            
        # qdot_h = mover.xdot2qdot(xdot_h)
        # qdot_h = qdot_h.tolist()

        # find current goal for human
        curr_goal = np.asarray(goals[goal_idx])
        if np.linalg.norm(curr_goal - q) < 0.1\
                 and goal_idx < len(goals)-1:
            goal_idx += 1

        qdot_h = get_human_action(curr_goal, q).tolist()

        curr_time = time.time()
        if curr_time - start_time >= step_time:
            traj.append(start_pos + s + qdot_h)
            start_time = curr_time

        if traj:
            # print(traj)
            # t = torch.Tensor(traj).unsqueeze(0)
            # print(t)
            alpha = model.classify([start_pos + s + qdot_h])
            # print(alpha[0])
            alphas.append(alpha[0])
        qdot = qdot_h
        # qdot = qdot[0]
        
        mover.send(qdot)
        rate.sleep()

def main():
    # model_name = [sys.argv[1]]
    # test_task = [sys.argv[2]]
    model_names = ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2",\
                      "open1", "open2"]
    test_tasks = ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2",\
                      "open1", "open2"]                      
    results = {}
    for model_name in model_names:
        alpha_per_task = []
        # print("Current model:", model_name)
        for test_task in test_tasks:
            alpha_mean = run_test(model_name, test_task)
            alpha_per_task.append(alpha_mean)
            print("model: {} task: {} confidence: {}".format(model_name, test_task, alpha_mean))
        results[model_name] = alpha_per_task
    print(results)
    results["description"] = "testing old classifier"
    results["trials"] = 1
    pickle.dump(results, open("model_confidences_old.pkl", "wb"))

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 