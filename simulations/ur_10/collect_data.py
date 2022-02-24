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
from sim_play_ur10 import Model
from std_msgs.msg import Float64MultiArray, String
from gen_data import tasks

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
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, stop

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
        return xyz

    def pose2joint(self, pose):
        return self.kdl_kin.inverse(pose, self.joint_states)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

    def send(self, qdot):
        self.qdots.append(qdot)
        qdot_mean = np.mean(self.qdots, axis=0).tolist()
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

def get_human_action(goal, state):
    noiselevel = 0.07
    action = (goal - state) * .5 + np.random.normal(0, noiselevel, len(goal))
    action = np.clip(action, -0.3, 0.3)
    return action

def compute_reward(goal, state):
    return -np.linalg.norm(goal-state)

def main(task, model_name):
    demo_num = "1"
    # task = sys.argv[1]
    # model_name = sys.argv[2]
    cae_model = 'models/' + 'cae_' + str(model_name)
    class_model = 'models/' + 'class_' + str(model_name)
    model = Model(class_model, cae_model)
    goals = tasks[task]
    rospy.init_node("data_collector")

    mover = TrajectoryClient()
    joystick = JoystickControl()
    reward = 0
    cum_reward = 0
    start_time = time.time()
    rate = rospy.Rate(1000)

    print("[*] Initialized, Moving Home")
    if np.linalg.norm(np.array(HOME) - np.array(mover.joint_states)) > 0.01:
        mover.switch_controller(mode='position')
        mover.send_joint(HOME, 5.0)
        mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")

    record = False
    flag1 = flag2 = True
    data = []
    steptime  = 0.4
    qdot_h = np.zeros(6)
    assist = False
    assist_start = 2.0
    start_q = mover.joint_states
    start_pos = mover.joint2pose()
    filename = "runs/" + model_name + "_" + task + "_" + demo_num + ".pkl"
    goal_idx = 0
    done = False
    while not rospy.is_shutdown():

        s_joint = mover.joint_states
        s = mover.joint2pose()
        # print(np.asarray(s).tolist() + np.asarray(start_q).tolist())
        t_curr = time.time() - start_time
        axes, start, mode, stop = joystick.getInput()
        start = True
        if stop or done or len(data)>=60:
            demonstration = {}
            demonstration["model"] = model_name
            demonstration["task"] = task
            demonstration["demo_num"] = demo_num
            demonstration["data"] = data
            if not data:
                print("[*] No data recorded")
            else:
                print(data[0])
            pickle.dump(demonstration, open(filename, "wb"))
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(data))
            mover.switch_controller(mode='position')
            mover.send_joint(s_joint, 1.0)
            return cum_reward

        if start and not record:
            record = True
            start_time = time.time()
            assist_time = time.time()
            print('[*] Recording the demonstration...')

        curr_time = time.time()
        if record and curr_time - assist_time >= assist_start and not assist:
            print("Assistance Started...")
            assist = True

        # compute which position the robot needs to go to
        if np.linalg.norm(np.asarray(goals[goal_idx]) - np.asarray(s_joint)) < 0.02:
            if goal_idx < len(goals)-1:
                goal_idx += 1
            else:
                done = True
        if record:
            qdot_h = get_human_action(np.array(goals[goal_idx]), np.array(s_joint))
            reward = compute_reward(np.array(goals[goal_idx]), np.array(s_joint))

        qdot_h = np.clip(qdot_h, -0.3, 0.3)

        alpha = model.classify(start_pos + s)
        alpha = np.clip(alpha, 0.0, 0.6)
        # alpha = 0.5
        z = model.encoder(start_pos + s)
        
        a_robot = model.decoder(z, s)
        a_robot = mover.xdot2qdot(a_robot)
        qdot_r = np.zeros(6)
        qdot_r = a_robot
        qdot_r = qdot_r.tolist()

        if assist:
            qdot = (alpha * 1.0 * np.asarray(qdot_r) + (1-alpha) * np.asarray(qdot_h))*2.0
            qdot = np.clip(qdot, -0.3, 0.3)
            qdot = qdot.tolist()
            qdot = qdot[0]
        else:
            qdot = qdot_h
            # qdot = qdot[0]

        if record and curr_time - start_time >= steptime:
            elapsed_time = curr_time - assist_time
            data.append([elapsed_time] + [s_joint] + [qdot_h.tolist()] + [qdot_r] + [float(alpha)] + [z] + [reward])
            start_time = curr_time
            print("qdot_h:{:2.4f} qdot_r:{:2.4f} alpha:{:2.4f} reward:{:2.4f}"\
                .format(np.linalg.norm(qdot_h), np.linalg.norm(qdot_r), float(alpha), float(reward)))
        cum_reward += reward
        mover.send(qdot)
        rate.sleep()

if __name__ == "__main__":
    # models = ['push1', 'push1_open1', 'push1_open1', 'push1_open1_scoop1', 'push1_open1_scoop1', 'push12_open1_scoop1_cut1',\
    #             'push12_open12_scoop1_cut1', 'push12_open12_scoop12_cut1', 'push12_open12_scoop12_cut12']
    models = ['push1_open1_scoop1']
    rewards = []
    for i in range(len(models)):
        task = "cut1"
        try:
            r = main(task, models[i])
            print(r)
            rewards.append(r)
        except rospy.ROSInterruptException:
            pass
    print(rewards)