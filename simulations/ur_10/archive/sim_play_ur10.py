#!/usr/bin/env python

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
import socket
import time
import random

import torch
import torch.nn as nn
from train_cae_ur10 import CAE
from train_classifier_ur10 import Net
import torch.nn.functional as F

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

HOME = [-1.571, -1.18997, -2.0167, -1.3992, 1.5407, 0.0]
END1 = [-1.571, -1.5710356871234339, -2.2512028853045862, -0.6734879652606409, 1.5406923294067383, 1.1984225238848012e-05]
END2 = [-1.571, -2.1604259649859827, -1.5543845335589808, -0.8909743467914026, 1.5406923294067383, 0.00022770027862861753]
# END1 = [-1.6811960379229944, -1.5710356871234339, -2.2512028853045862, -0.6734879652606409, 1.5406923294067383, 1.1984225238848012e-05]
# END2 = [-1.5708683172809046, -2.1604259649859827, -1.5543845335589808, -0.8909743467914026, 1.5406923294067383, 0.00022770027862861753]
# END1 = [-0.800208870564596, -1.789145294819967, -2.005460564290182, -0.8623107115374964, 1.4747997522354126, 0.769659161567688]
# END2 = [-1.0631392637835901, -2.302997414265768, -1.2114885489093226, -1.119192902241842, 1.4926565885543823, 0.5063522458076477]

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100
device = "cpu"


class Model(object):

    def __init__(self, classifier_name, cae_name):
        self.class_net = Net()
        self.cae_net = CAE()
        
        model_dict = torch.load(classifier_name, map_location='cpu')
        self.class_net.load_state_dict(model_dict)
        
        model_dict = torch.load(cae_name, map_location='cpu')
        self.cae_net.load_state_dict(model_dict)

        self.class_net.eval
        self.cae_net.eval

    def classify(self, c):
        labels = self.class_net.classifier(torch.FloatTensor(c))
        confidence = F.softmax(labels, dim=0)
        return confidence.data[0].numpy()

    def encoder(self, c):
        z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
        return z_mean_tensor.tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.cae_net.decoder(z_tensor)
        return a_predicted.data.numpy()


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
        z2 = -self.gamepad.get_axis(1)
        z3 = -self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * -z[1], STEP_SIZE_A * -z[0], STEP_SIZE_A * -z[2])
        else:
            self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * z[2], 0, 0, 0)

class TrajectoryClient(object):

    def __init__(self):
        # Action client for joint move commands
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        # Velocity commands publishSTEP_SIZE_Ler
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
        self.cartesian_pose = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        
        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
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


    def joint2pose(self):
        state = self.kdl_kin.forward(self.joint_states)
        xyz_lin = np.array(state[:,3][:3]).T
        xyz_lin = xyz_lin.tolist()
        R = state[:,:3][:3]
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = xyz = np.asarray(xyz_lin[-1]).tolist() + np.asarray(xyz_ang).tolist()

        return xyz#self.kdl_kin.forward(self.joint_states)

    def pose2joint(self, pose):
        return self.kdl_kin.inverse(pose, self.joint_states)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return np.dot(J_inv, xdot)

    def send(self, xdot):
        qdot = xdot#self.xdot2qdot(xdot)
        # self.qdots.append(qdot)
        # qdot_mean = np.mean(self.qdots, axis=0).tolist()[0]
        cmd_vel = Float64MultiArray()
        cmd_vel.data = qdot
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


class RecordClient(object):

    def __init__(self):
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.joint_states = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        self.script_pub = rospy.Publisher('/ur_hardware_interface/script_command', \
                                            String, queue_size=100)

        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

    def joint_states_cb(self, msg):
        try:
            states = list(msg.position)
            states[2], states[0] = states[0], states[2]
            self.joint_states = tuple(states)
        except:
            pass

    def send_cmd(self, cmd):
        self.script_pub.publish(cmd)

    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()


def main():
    # demo_num = sys.argv[1]
    rospy.init_node("teleop")

    mover = TrajectoryClient()
    joystick = JoystickControl()
    recorder = RecordClient()

    start_time = time.time()
    rate = rospy.Rate(1000)

    print("[*] Initialized, Moving Home")
    mover.switch_controller(mode='position')
    mover.send_joint(HOME, 5.0)
    mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")


    demos = sys.argv[1]
    cae_model = 'models/' + 'cae_' + str(demos)
    class_model = 'models/' + 'class_' + str(demos)
    model = Model(class_model, cae_model)

    record = False
    flag = True
    demonstration = []
    data = []
    steptime  = 0.1
    scaling_trans = 0.1
    scaling_rot = 0.2
    assist = False
    assist_start = 5.0
    action = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    start_q = mover.joint2pose()
    # qdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # filename = "demo" + demo_num + ".pkl"

    while not rospy.is_shutdown():

        s_joint = np.asarray(mover.joint_states).tolist()
        s = mover.joint2pose()
        # print(np.asarray(s).tolist() + np.asarray(start_q).tolist())
        t_curr = time.time() - start_time
        axes, start, mode, stop = joystick.getInput()
        if stop:
            # pickle.dump(demonstration, open(filename, "wb"))
            # print(demonstration)
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            mover.switch_controller(mode='position')
            mover.send_joint(s_joint, 1.0)
            return True

        if start and not record:
            record = True
            start_time = time.time()
            assist_time = time.time()
            print('[*] Recording the demonstration...')

        xdot_h = np.zeros(6)
        if mode:
            xdot_h[3:] = scaling_trans * np.asarray(axes)
        else:
            xdot_h[:3] = scaling_trans * np.asarray(axes)
            
        qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()
        # if np.linalg.norm(np.asarray(END1) - np.asarray(s_joint)) > 0.02 and flag and record:
        #     action = (np.asarray(END1) - np.asarray(s_joint))*0.25
        #     action = np.clip(action, -0.3, 0.3)
        # elif record:
        #     action = (np.asarray(END2) - np.asarray(s_joint))*0.25
        #     action = np.clip(action, -0.3, 0.3)
        #     flag = False
        # qdot_h = action*0.5
        # # print(qdot_h)
        # qdot_h = qdot_h.tolist()

        alpha = model.classify(start_q + s + qdot_h[0])
        alpha = np.clip(alpha, 0.0, 0.6)
        # alpha = 0.5
        z = model.encoder(start_q + s)
        
        a_robot = model.decoder(z, s)
        a_robot = mover.xdot2qdot(a_robot)
        qdot_r = np.zeros(6)
        qdot_r = 1*a_robot
        qdot_r = qdot_r.tolist()


        curr_time = time.time()
        if record and curr_time - assist_time >= assist_start and not assist:
            print("Assistance Started...")
            assist = True
            demonstration.append(np.asarray(start_q).tolist() + np.asarray(s).tolist())
            
            start_time = curr_time

        if assist:
            alpha = 0
            qdot = (alpha * 1.0 * np.asarray(qdot_r) + (1-alpha) * np.asarray(qdot_h))*2.0
            qdot = np.clip(qdot, -0.3, 0.3)
            qdot = qdot.tolist()
            qdot = qdot[0]
        else:
            qdot = qdot_h
            qdot = qdot[0]

        # qdot = np.clip(qdot, -0.2,0.2)

        if record and curr_time - start_time >= steptime:
            # print(s_cart)
            # print(s_cart[:,:3][:3])
            demonstration.append(start_q + s)
            elapsed_time = curr_time - assist_time
            qdot_h = qdot_h
            qdot_r = qdot_r
            data.append([elapsed_time] + [s] + [qdot_h] + [qdot_r] + [float(alpha)])
            # print(z)
            start_time = curr_time
            print(float(alpha))
            # print("qdot = {}, qdot_r = {}" .format(qdot,qdot_r))
      
        # joystick.getAction(axes)
        # action = np.array([-0.1,0.0,0.0,0.0,0.0,0.0])
        # mover.send(action)
        
        # print(qdot)
        
        mover.send(qdot)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 