import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from model_gcl import CostNN
from utils import to_one_hot, get_cumulative_rewards, TrajectoryClient, convert_to_6d
from torch.optim.lr_scheduler import StepLR
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
import rospy, argparse
from glob import glob
from geometry_msgs.msg import Twist
from gcl import GCL
from waypoints import HOME
import time 
from replay_memory import ReplayMemory
from math import pi
from torch.utils.tensorboard import SummaryWriter
import datetime
from mpl_toolkits.mplot3d import Axes3D
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import (
    Robotiq2FingerGripperDriver as Robotiq
)


class RobotInterface():
    def __init__(self):
        self.rate = rospy.Rate(10)
        self.mover = TrajectoryClient()
        self.home_pose = self.mover.joint2pose(HOME)
        self.home_xyz = self.home_pose[0:3]
        rospy.loginfo("Initialized, Moving Home")
        self.mover.go_home()
        self.mover.reset_gripper()
        self.trans_mode = 1
        self.slow_mode = 0
        self.gripper_mode = 0 # gripper not actuated at start, is open
        self.trans_mode_last_update = time.time()
        self.gripper_mode_last_update = time.time()
        self.slow_mode_last_update = time.time()

    def goHome(self):
        x = self.mover.go_home()
        self.mover.reset_gripper()
        return x

    def rotLimit(self, rot):
        # need to be sure to not exceed any joint limits. Say 
        # limit the net roll to < -pi/ 2 OR > pi / 2, since it goes from pi to -pi at home
        return rot < -pi / 2 or rot > pi / 2

    # action space is expected to be a dict with the following:
    # axes: (3,)
    # trans_mode, slow_mode, gripper_ac
    def applyActionOLD(self, action):
        if type(action) == dict:
            axes = action["axes"]
            gripper = action["gripper_ac"]
            trans_mode = action["trans_mode"]
            slow_mode = action["slow_mode"]
        else: 
            axes = action[0:3]
            roll = action[3]
            trans_mode = action[4]
            gripper = action[5]
            slow_mode = action[6]
        gripper_ac = 0
        if gripper and (self.mover.robotiq_joint_state > 0):
            self.mover.actuate_gripper(gripper_ac, 1, 0.) # async?
        
        elif gripper and (self.mover.robotiq_joint_state == 0):
            gripper_ac = 1
            self.mover.actuate_gripper(gripper_ac, 1, 0) # is this async?
 
        else: 
            scaling_trans = 0.2 - 0.1*slow_mode 
            scaling_rot = 0.4 - 0.2*slow_mode 
            
            xdot_h = np.zeros(6)
            xdot_h[:3] = scaling_trans * np.asarray(axes)
            xdot_h[3] = scaling_rot * roll
            # if trans_mode: 
            #     xdot_h[:3] = scaling_trans * np.asarray(axes)

            # elif not trans_mode:
            #     # change coord frame from robotiq gripper to tool flange
            #     R = np.mat([[1, 0, 0],
            #                 [0, 1, 0],
            #                 [0, 0, 1]])
            #     P = np.array([0, 0, -0.10])
            #     P_hat = np.mat([[0, -P[2], P[1]],
            #                     [P[2], 0, -P[0]],
            #                     [-P[1], P[0], 0]])
                
            #     axes = np.array(axes)[np.newaxis]
            #     trans_vel = scaling_rot * P_hat * R * axes.T
            #     rot_vel = scaling_rot * R * axes.T
            #     xdot_h[:3] = trans_vel.T[:]
            #     xdot_h[3:] = rot_vel.T[:]
                
            qdot_h = self.mover.xdot2qdot(xdot_h)
            qdot_h = qdot_h.tolist()
            
            qdot_h = self.mover.compute_limits(qdot_h)

            self.mover.send(qdot_h[0])
            # ctime = time.time() # hold command for 100 ms
            # while time.time() < ctime + 0.10:
            #     if self.isDone():
            #         qdot_h = self.mover.xdot2qdot([0, 0, 0, 0, 0, 0])
            #         qdot_h = qdot_h.tolist()    
            #         self.mover.send(qdot_h[0])
        self.rate.sleep()
        return self.getState(), self.isDone() 

    # slow mode and trans mode vary from -1 to 1. if > 0, ==> 1
    def applyAction(self, action, flagUpdateTime=1, gripperUpdateTime=3):
        xdot_h = action[0:6]
        slow_mode = action[6]
        trans_mode = action[7]
        gripper_mode = action[8]
        slow_mode = float(slow_mode > 0.0)
        trans_mode = float(trans_mode > 0.0)
        gripper_mode = float(gripper_mode > 0.0)
        if slow_mode != self.slow_mode and \
            time.time() > self.slow_mode_last_update + flagUpdateTime:
            # update slow mode
            self.slow_mode = slow_mode
            self.slow_mode_last_update = time.time()
        if trans_mode != self.trans_mode and \
            time.time() > self.trans_mode_last_update + flagUpdateTime:
            # update trans mode
            self.trans_mode = trans_mode
            self.trans_mode_last_update = time.time()
        if gripper_mode != self.gripper_mode and \
            time.time() > self.gripper_mode_last_update + gripperUpdateTime:
            # update gripper mode
            self.gripper_mode = gripper_mode
            Robotiq.goto(self.mover.robotiq_client, 
                pos=1-gripper_mode, speed=1, force=0, block=False) # TODO fix
            # this is ugly
            self.gripper_mode_last_update = time.time()
        # note that these actions do not have to affect the world, they
        # just have to affect the robot's perception of it 
        qdot_h = self.mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()
            
        qdot_h = self.mover.compute_limits(qdot_h)

        self.mover.send(qdot_h[0])
        self.rate.sleep()

        return self.getState(), self.isDone()

    def stall(self):
        qdot_h = self.mover.xdot2qdot(self.getState()[-6:-3])
        self.mover.send(qdot_h.tolist()[0])
        return

    def isDone(self):
        s = self.getState()
        xyz = s[6:9]
        rpy = s[12:15]
        q4 = s[3]
        outsideArea = np.linalg.norm(np.subtract(xyz, self.home_xyz)) > (self.home_xyz[-1] - 0.15)
        nearBase = np.linalg.norm(np.subtract(xyz, [.05, 0.2, 0.42])) < 0.125
        closeToWires = xyz[0] > 0.52
        tooHigh = xyz[2] > 0.8
        tooLow = xyz[2] < 0.24
        limitY = xyz[1] < 0.31
        limitJoint4 = q4 > -0.7
        # return outsideArea or nearBase or closeToWires or tooHigh or limitY or tooLow or not self.rotLimit(rpy[0])
        return outsideArea or nearBase or closeToWires or tooHigh or limitY or tooLow or limitJoint4


    def getState(self):
        q = self.mover.joint_states
        curr_pos = convert_to_6d(self.mover.joint2pose())
        c_gripper_pos = self.mover.robotiq_joint_state
        trans_mode = self.trans_mode
        slow_mode = self.slow_mode
        z = list(q) + list(curr_pos) + [c_gripper_pos] + [trans_mode] + [slow_mode]
        return z
    # observe new states and return
    def getStateOLD(self):
        q = self.mover.joint_states
        curr_pos = self.mover.joint2pose()
        return (list(q) + list(curr_pos))
    
    def randomAction(self):
        return np.concatenate((np.random.uniform(-0.4, 0.4, 6), 
            np.random.uniform(-1, 1, 3))) # xdot_h and different modes

    def randomActionOLD(self):
        return np.concatenate((np.random.uniform(-1., 1., 3), np.random.uniform(-0.5, 0.5, 1)))

    
# learns an intent using GCL given expert demonstrations from an intentFolder
# saves the policy to intents/{intentFolder}/ and cost to intents/{intentFolder}/cost
def learn_intent(args):
    inpIntentFolder = args.intentFolder
    robot_interface = RobotInterface()
    intentFolder = "intent" + str(inpIntentFolder)
    

    # demo_trajs = pickle.load(open("intent_samples/"+intentFolder+"/all_actions_simple.pkl", "r"))
    # state_shape = (len(demo_trajs[0][0]),) # demo trajs = (state, action), size of the state space

    # state_shape should be (12,)
    # n_actions = len(demo_trajs[0][1]) # size of the action space
    action_size = 9 # xdot_h + gripper_ac + slow_mode + trans_mode
    init_state = robot_interface.getState()
    state_size = len(init_state)

    # INITILIZING POLICY AND REWARD FUNCTION
    agent = GCL(action_dim=action_size, state_dim=state_size)
    # Tensorboard
    folder = "runs/gcl/intent1/"
    writer = SummaryWriter(folder + '{}'.format(datetime.datetime.now().strftime("%m-%d_%H-%M")))


    if args.resume:
        agent.load_model("gcl", args.resume_ep)

    # Memory
    memory_novice = ReplayMemory(capacity=args.memory_size)
    memory_expert = ReplayMemory(capacity=args.memory_size)
    memory_expert.load_buffer("expert", args.expert)
    print(len(memory_expert))
    args.batch_size = min(args.batch_size, len(memory_expert))
    
    # Main loop
    for i_episode in range(1, args.num_eps+1):
        # trajs = [robot_interface.generate_session(t_max=2000)  for _ in range(EPISODES_TO_PLAY)]
        # sample_trajs = trajs + sample_trajs

        episode_reward = 0
        episode_steps = 0
        state = robot_interface.getState()
        done = False # robot_interface.isDone()
        outOfBounds = False

        STALL_LEN = 2
        while not done:
            outOfBounds = outOfBounds or robot_interface.isDone()
            if outOfBounds and STALL_LEN > 0:
                STALL_LEN -= 1
                robot_interface.mover.stop()
                rospy.loginfo("Stopped robot")

            if i_episode < args.start_eps:
                action = robot_interface.randomAction()
                # action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # update score function
            if len(memory_novice) > args.batch_size:
                cost_loss = agent.update_cost(memory_expert, memory_novice, args.batch_size)
                writer.add_scalar('cost/loss', cost_loss, agent.cost_updates)

            # update policy
            if len(memory_novice) > args.batch_size:
                critic_1_loss, critic_2_loss, policy_loss = agent.update_policy(memory_novice, args.batch_size)
                writer.add_scalar('SAC/critic_1', critic_1_loss, agent.policy_updates)
                writer.add_scalar('SAC/critic_2', critic_2_loss, agent.policy_updates)
                writer.add_scalar('SAC/policy', policy_loss, agent.policy_updates)
           
            reward = 0 # TODO
            episode_steps += 1
            episode_reward += reward
            done = episode_steps > args.max_episode_steps
            mask = int(done)

            if not outOfBounds:
                # print(action)
                next_state, outOfBounds = robot_interface.applyAction(action)
            # need to transform action to be integers for the modes?
        
            memory_novice.push(state, action, reward, next_state, mask)
      
            state = next_state
        # STALL_LEN = 10
        # for i in range(STALL_LEN):
        #     robot_interface.applyAction([0, 0, 0, 0, 1, 0, 1])
        robot_interface.mover.stop()
        robot_interface.goHome()
        writer.add_scalar('reward/episode_reward', episode_reward, i_episode)
        print("\nEpisode: {}, reward: {}, final state: {}".format(i_episode, round(episode_reward, 2), state))

        if i_episode % 5 == 0:
            agent.save_model("gcl", str(i_episode) + str(intentFolder))
            memory_novice.save_buffer("gcl", str(i_episode) + str(intentFolder))



    # for i in range(1000):
    #     trajs = [policy.generate_real_session(robot_interface, t_max=2000)  for _ in range(EPISODES_TO_PLAY)]
    #     sample_trajs = trajs + sample_trajs

       


if __name__ == "__main__":
    rospy.init_node("train_intent")
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert', default=0)
    parser.add_argument("--intentFolder", type=int, help="intent folder to read from and save to", default=0)
    parser.add_argument("--generate", type=bool, help="whether to (re)generate the intent dataset for intentFolder", default=False)
    parser.add_argument('--memory_size', type=int, default=50000)
    parser.add_argument('--num_eps', type=int, default=500)
    parser.add_argument('--start_eps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--max_episode_steps', type=int, default=500)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_ep', default='')
    args = parser.parse_args()
    learn_intent(args)




    # notes: run 15:19 on Nov 3 is first run with roll and steps=1000