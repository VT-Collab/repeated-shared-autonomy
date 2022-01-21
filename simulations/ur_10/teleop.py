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

HOME = [-1.571, -1.2609770933734339, -1.842959229146139, \
        0.07699000835418701, 1.6361737251281738, 0.0]

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100

class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(1)
        z2 = -self.gamepad.get_axis(0)
        z3 = self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        gripper_open = self.gamepad.get_button(1)
        gripper_close = self.gamepad.get_button(0)
        return tuple(z), (gripper_open, gripper_close), stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * -z[1], STEP_SIZE_A * -z[0], STEP_SIZE_A * -z[2])
        else:
            self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * -z[2], 0, 0, 0)

class TrajectoryClient(object):

    def __init__(self):
        # Action client for joint move commands
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        # Velocity commands publisher
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',\
                 Float64MultiArray, queue_size=10)
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        # service call to switch controllers
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller',\
                 SwitchController)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.joint_states = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        6361737251281738
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

        # store previous joint vels for moving avg
        self.qdots = deque(maxlen=MOVING_AVERAGE)
        for idx in range(MOVING_AVERAGE):
            self.qdots.append(np.asarray([0.0] * 6))

    def joint_states_cb(self, msg):
        try:
            if msg is not None:
                states = list(msg.position)
                states[2], states[0] = states[0], states[2]
                self.joint_states = tuple(states) 
        except:
            pass
    
    def switch_controller(self, mode=None):
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        if mode == 'velocity':
            req.start_controllers = ['joint_group_vel_controller']
            req.stop_controllers = ['scaled_pos_joint_traj_controller']
            req.strictness = req.STRICT
        elif mode == 'position':
            req.start_controllers = ['scaled_pos_joint_traj_controller']
            req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = req.STRICT
        else:
            rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

    def send(self, xdot):
        qdot = self.xdot2qdot(xdot)
        self.qdots.append(qdot)
        qdot_mean = np.mean(self.qdots, axis=0).tolist()[0]
        cmd_vel = Float64MultiArray()
        cmd_vel.data = qdot_mean
        self.vel_pub.publish(cmd_vel)

    def send_joint(self, pos, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = pos
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()

def main():
    rospy.init_node("teleop")

    mover = TrajectoryClient()
    joystick = JoystickControl()

    start_time = time.time()
    rate = rospy.Rate(1000)

    print("[*] Initialized, Moving Home")
    mover.switch_controller(mode='position')
    mover.send_joint(HOME, 4.0)
    mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")

    while not rospy.is_shutdown():
        t_curr = time.time() - start_time
        axes, buttons, stop = joystick.getInput()
        print(axes)
        if stop:
            #pickle.dump(data, open(filename, "wb"))
            return True
      
        joystick.getAction(axes)
        mover.send(joystick.action)
        if joystick.A_pressed:
            mover.actuate_gripper(0.05, 0.1, 10)
        elif joystick.B_pressed:
            mover.actuate_gripper(0., 0.1, 10)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass