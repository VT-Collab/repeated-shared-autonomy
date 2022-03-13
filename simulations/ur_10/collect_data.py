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

# Imports from current directory
from utils import TrajectoryClient, JoystickControl, Model, get_human_action, compute_reward, go2home
from tasks import HOME, TASKSET 

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 10



def main(model_name, max_runs):
    
    rospy.init_node("data_collector")
    rate = rospy.Rate(1000)

    mover = TrajectoryClient()
    joystick = JoystickControl()

    cae_model = 'models/' + 'cae_' + "_".join(model_name) + "_old"
    class_model = 'models/' + 'class_' + "_".join(model_name) + "_old"
    model = Model(class_model, cae_model)
    
    while not rospy.is_shutdown():
        reward_per_task = []
        for task in model_name:
            reward_per_run = []
            for run in range(1, max_runs+1):
                go2home()
                # rospy.sleep(2)            
                filename = "runs/" + "model_" + "_".join(model_name) + "_task_"\
                             + task + "_" + str(run) + ".pkl"

                reward = 0
                cum_reward = 0
                data = []
                steptime  = 0.1
                assist = False
                assist_start = 3.0
                start_q = mover.joint_states
                start_pos = mover.joint2pose()
                goals = TASKSET[task]
                goal_idx = 0
                qdot_h = [0] * 6
                qdor_r = [0] * 6
                done = False
                start_time = time.time()
                assist_time = time.time()

                while not done:
                    q = np.asarray(mover.joint_states).tolist()
                    s = mover.joint2pose()
                    axes, start, mode, stop = joystick.getInput()

                    if stop:
                        mover.switch_controller(mode='position')
                        mover.send_joint(q, 1.0)
                        return None

                    curr_time = time.time()
                    if curr_time - assist_time >= assist_start and not assist:
                        assist = True

                    # compute which position the robot needs to go to
                    curr_goal = np.asarray(goals[goal_idx])
                    if np.linalg.norm(curr_goal - q) < 0.15\
                             and goal_idx < len(goals)-1:
                        goal_idx += 1
                    
                    qdot_h = get_human_action(np.array(goals[goal_idx]), np.array(q))
                    reward = compute_reward(np.array(goals[goal_idx]), np.array(q))

                    alpha = 0
                    if s:
                        alpha = model.classify(start_pos + s + qdot_h.tolist())
                        alpha = min(alpha, 0.6)
                        z = model.encoder(start_pos + s)                
                        a_robot = model.decoder(z, s)
                        a_robot = mover.xdot2qdot(a_robot)
                        qdot_r = a_robot
                        qdot_r = qdot_r.tolist()

                    if assist:
                        # alpha = .8
                        # qdot = (alpha * 2.5 * np.asarray(qdot_r) + (1-alpha) * np.asarray(qdot_h))#*2.0
                        qdot = (0.8 * 2.5 * np.asarray(qdot_r) + 0.2 * np.asarray(qdot_h))#*2.0
                        qdot = np.clip(qdot, -0.1, 0.1)
                        qdot = qdot.tolist()[0]
                    else:
                        qdot = qdot_h

                    if curr_time - start_time >= steptime:
                        elapsed_time = curr_time - assist_time
                        data.append([elapsed_time] + [q] + [qdot_h.tolist()] + [qdot_r] + [float(alpha)] + [z] + [reward])
                        start_time = curr_time
                        # print("model: {0} task: {1} run: {2} qdot_h:{3:2.1f} qdot_r:{4:2.1f} alpha:{5:2.1f} reward:{6:2.1f}"\
                        #     .format("_".join(model_name), task, run, np.linalg.norm(qdot_h), np.linalg.norm(qdot_r), float(alpha),\
                        #      float(reward)))
                        cum_reward += reward

                    if len(data) >= 155:
                        cum_reward -= reward
                        reward = 50 * compute_reward(np.array(goals[-1]), np.array(q))
                        data[-1][-1] = reward
                        cum_reward += reward
                        print("model: {0} task: {1} cum_reward: {2:2.1f}".format("_".join(model_name), task, cum_reward))
                        # print(50*compute_reward(np.array(goals[-1]), np.array(q)))
                        # print(data[-1][-1], reward)
                        # reward = -50 * (len(goals) - goal_idx - 1)
                        # print(len(goals), goal_idx)
                        # print(data[-1][-1], reward)
                        done = True

                    qdot = mover.compute_limits(qdot)
                    
                    if done:
                        mover.switch_controller(mode='position')
                        mover.send_joint(q, 1.0)
                    else:
                        mover.send(qdot[0])
                    rate.sleep()

                reward_per_run.append(cum_reward)
                demonstration = {}
                demonstration["model"] = "_".join(model_name)
                demonstration["task"] = task
                demonstration["run"] = run
                demonstration["data"] = data
                # if not data:
                #     print("[*] No data recorded")
                # else:
                #     print(data[0])
                pickle.dump(demonstration, open(filename, "wb"))
                mover.switch_controller(mode='position')
                mover.send_joint(q, 1.0)
            reward_per_task.append(float(np.mean(reward_per_run)))
        return reward_per_task

if __name__ == "__main__":
    model_names = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
                      ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
                      ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
                      ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]

    max_runs = 5
    rewards = {}
    for model_name in model_names:
        try:
            r = main(model_name, max_runs)
            if not r:
                break
            else:
                print(r)
                rewards["_".join(model_name)] = r
        except rospy.ROSInterruptException:
            pass
        print(rewards)
        pickle.dump(rewards, open("rewards_old.pkl", "wb"))