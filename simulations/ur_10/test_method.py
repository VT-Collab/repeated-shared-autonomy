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
from std_msgs.msg import Float64MultiArray, String

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
from utils import TrajectoryClient, JoystickControl, Model, get_human_action, compute_reward
from tasks import HOME, TASKSET 


def main():
    rospy.init_node("test_method")

    mover = TrajectoryClient()
    joystick = JoystickControl()

    start_time = time.time()
    rate = rospy.Rate(1000)

    print("[*] Initialized, Moving Home")
    if np.linalg.norm(np.array(HOME) - np.array(mover.joint_states)) > 0.01:
        mover.switch_controller(mode='position')
        mover.send_joint(HOME, 5.0)
        mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")


    model_name = sys.argv[1]
    # cae_model = 'models/' + 'cae_' + str(model_name)
    class_model = 'models/' + 'class_' + str(model_name)
    model = Model(class_model)#, cae_model)

    step_time  = 0.4
    scaling_trans = 0.1
    scaling_rot = 0.2
    start_q = mover.joint2pose()
    goals = TASKSET["push1"]
    goal_idx = 0
    traj = []

    while not rospy.is_shutdown():

        q = np.asarray(mover.joint_states).tolist()
        s = mover.joint2pose()
        axes, start, mode, stop = joystick.getInput()
        if stop:
            # pickle.dump(demonstration, open(filename, "wb"))
            # # print(demonstration)
            # print("[*] Done!")
            # print("[*] I recorded this many datapoints: ", len(demonstration))
            mover.switch_controller(mode='position')
            mover.send_joint(q, 1.0)
            return True

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

        qdot = get_human_action(curr_goal, q)

        curr_time = time.time()
        if curr_time - start_time >= step_time:
            traj.append(start_q + s + qdot_h[0])
            start_time = curr_time

        if traj:
            # print(traj)
            t = torch.Tensor(traj).unsqueeze(0)
            # print(t)
            alpha = model.classify(t)
            print(alpha)
            
        qdot = qdot_h
        qdot = qdot[0]
        
        mover.send(qdot)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 