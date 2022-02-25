# Standard imports
import rospy, actionlib, copy
import sys, time, pygame, pickle
import numpy as np
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from collections import deque
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
from utils import TrajectoryClient, JoystickControl
from tasks import HOME, TASKSET 

def main():
	rospy.init_node("record_demos")
	r = TrajectoryClient()

if __name__ == "__main__":
        try:
            main()
        except rospy.ROSInterruptException:
            pass