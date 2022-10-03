#!/usr/bin/env python
import rospy
import actionlib
import numpy as np
import pygame
import pickle
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
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

from waypoints import HOME

"""
TODO
- Clean up global variables
- Get active controller before switching
"""

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
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [-z1, z2, -z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        X_pressed = self.gamepad.get_button(2)
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, X_pressed, stop

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
                 Float64MultiArray, queue_size=100)
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

        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        self.robotiq_sub = rospy.Subscriber('/robotiq_joint_state', JointState, self.robotiq_joint_state_cb)
        self.robotiq_joint_state = None
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
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
    
    def robotiq_joint_state_cb(self, msg):
        try:
            if msg is not None:
                self.robotiq_joint_state = msg.position[0]
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

    def go_home(self):
        # Sometimes going home fails because joint_states are None
        while True:
            try:
                if np.linalg.norm(np.array(HOME) - np.array(self.joint_states)) > 0.01:
                    if np.linalg.norm(np.array(HOME) - np.array(self.joint_states)) < 1.5:
                        time = 5.
                    else:
                        time = 8.
                    self.switch_controller(mode='position')
                    self.send_joint(HOME, time)
                    self.client.wait_for_result()
                self.switch_controller(mode='velocity')
            except:
                continue
            break
        return True  

    def reset_gripper(self):
        Robotiq.goto(self.robotiq_client, pos=1, speed=1, force=0, block=True)

    def joint2pose(self, q = None):
        if q is None:
            q = self.joint_states
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

    def pose2joint(self, pose, guess=None):
        if guess is None:
            guess = self.joint_states
        return self.kdl_kin.inverse(pose, guess, maxiter=1000)

    def qdot2xdot(self, qdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        return J.dot(qdot)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

    def compute_limits(self, qdot, type="table"):
        if isinstance(qdot[0], list):
            xdot = self.qdot2xdot(qdot[0])
        else:
            xdot = self.qdot2xdot(qdot)
        # print(xdot)
        s_next = self.joint2pose() + 0.1 * xdot
        # if s_next[0,2] < 0.35:
        #     xdot[0,3] = 0.
        #     xdot[0,4] = 0.
        #     xdot[0,5] = 0.
        if s_next[0,2] <= 0.1:
            # print("close to table")
            xdot[0,2] = 0.
        # if s_next[0,0] > 0.45 or s_next[0,0] < -0.9:
        #     # print("edited x")
        #     xdot[0,0] = 0
        # if s_next[0,1] < 0.23 or s_next[0,1] > 1.12:
        #     # print("edited y")
        #     xdot[0,1] = 0
        qdot = self.xdot2qdot(xdot.transpose()).transpose()
        # if self.joint_states[2] > -0.7 and qdot[0, 2] > 0:
        #     # print("edited qdot")
        #     qdot[0, 2] = 0.
        qdot = qdot.tolist()        
        return qdot

    def send(self, qdot):
        self.qdots.append(qdot)
        qdot_mean = np.mean(self.qdots, axis=0).tolist()
        # qdot_mean = self.compute_limits(qdot_mean)[0]
        # print(qdot_mean)
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
        # Need to invert when sending commands. 
        # For Robotiq.goto pos=1 -> close and pos=0 -> open
        pos = 1 - pos
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()

def deform(xi, start, length, tau):
    # length += np.random.choice(np.arange(30, length))
    xi1 = np.asarray(xi).copy()
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(np.dot(A.T, A))
    U = np.zeros(length)
    gamma = np.zeros((length, 6))
    for idx in range(6):
        U[0] = tau[idx]
        gamma[:,idx] = np.dot(R, U)
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end+1,:] += gamma[0:end-start+1,:]
    return xi1

def get_rotation_mat(euler):
    
    R_x = np.mat([[1, 0, 0],
                  [0, np.cos(euler[0]), -np.sin(euler[0])],
                  [0, np.sin(euler[0]), np.cos(euler[0])]])

    R_y = np.mat([[np.cos(euler[1]), 0, np.sin(euler[1])],
                  [0, 1, 0],
                  [-np.sin(euler[1]), 0, np.cos([1])]])

    R_z = np.mat([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                  [np.sin(euler[2]), np.cos(euler[2]), 0],
                  [0, 0, 1]])
    R = R_x * R_y * R_z
    return R

def convert_to_6d(pos):
    pos_awrap = np.zeros(9)
    pos_awrap[:3] = pos[:3]
    pos_awrap[3:] = get_rotation_mat(pos[3:]).flatten('F')[0,:6]
    return pos_awrap
