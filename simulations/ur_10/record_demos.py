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
from utils import TrajectoryClient, JoystickControl, get_human_action, compute_reward
from tasks import HOME, TASKSET 

def generate_trajectory(task, demo_num):
    rospy.init_node("record_demo")
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

    demo = []
    step_time = 0.1
    qdot = np.zeros(6)
    start_pos = mover.joint2pose()
    goals = TASKSET[task]
    goal_idx = 0

    print("[*] Task: {} Demo: {}".format(task, demo_num))
    while not rospy.is_shutdown():

        q = mover.joint_states
        curr_pos = mover.joint2pose()

        axes, start, mode, stop = joystick.getInput()

        if stop or len(demo) >= 250:
            print("[*] Datapoints in trajectory: ", len(demo))
            mover.switch_controller(mode='position')
            mover.send_joint(q, 1.0)
            return demo

        # find current goal for human
        curr_goal = np.asarray(goals[goal_idx])
        if np.linalg.norm(curr_goal - q) < 0.1\
                 and goal_idx < len(goals)-1:
            goal_idx += 1

        qdot = get_human_action(curr_goal, q)
        reward = compute_reward(curr_goal, q)

        curr_time = time.time()
        if curr_time - start_time >= step_time:
            demo.append([start_pos + curr_pos, qdot.tolist()])
            start_time = curr_time

        mover.send(qdot)
        rate.sleep()

def main():
    max_demos = 15
    for task in TASKSET:
        for demo_num in range(11,max_demos+1):
            savename = "demos/" + task + "_" + str(demo_num) + ".pkl"
            demo = generate_trajectory(task, demo_num)
            print(demo)
            pickle.dump(demo, open(savename, "wb"))
if __name__ == "__main__":
        try:
            main()
        except rospy.ROSInterruptException:
            pass