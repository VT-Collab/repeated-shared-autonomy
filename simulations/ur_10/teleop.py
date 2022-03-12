#!/usr/bin/env python

import rospy
import actionlib
import sys
import time
import numpy as np
import pygame
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import copy
from collections import deque

from std_msgs.msg import Float64MultiArray

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
from tasks import HOME, TASKSET 

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100

from utils import TrajectoryClient, JoystickControl, go2home

def main():
    rospy.init_node("teleop")

    scaling_trans = 0.1
    mover = TrajectoryClient()
    joystick = JoystickControl()

    start_time = time.time()
    rate = rospy.Rate(1000)

    print("[*] Initialized, Moving Home")
    mover.switch_controller(mode='position')
    if np.linalg.norm(np.array(HOME) - np.array(mover.joint_states)) > 0.01:
        mover.switch_controller(mode='position')
        mover.send_joint(HOME, 5.0)
        mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")

    while not rospy.is_shutdown():
        t_curr = time.time() - start_time
        axes, start, mode, stop = joystick.getInput()
        q = mover.joint_states
        s = mover.joint2pose()
        # print(axes)
        if stop:
            #pickle.dump(data, open(filename, "wb"))
            return True
      
        xdot_h = np.zeros(6)
        if mode:
            xdot_h[3:] = scaling_trans * np.asarray(axes)
            # print(xdot_h)
        else:
            xdot_h[:3] = scaling_trans * np.asarray(axes)
        # print(xdot_h)
            
        qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()
        
        qdot_h = mover.compute_limits(qdot_h)
        mover.send(qdot_h[0])
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass