import time
import numpy as np
import pickle
import pygame
import sys
import random
import os
import torch
import torch.nn as nn
from env import SimpleEnv
from train_cae import CAE
from train_classifier import Net
import torch.nn.functional as F
import pybullet as p

HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
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


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1
        self.timeband = 0.5
        self.lastpress = time.time()
        if os.name == "posix":
            self.z_axis = 3
        else:
            self.z_axis = 4

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        dx = self.gamepad.get_axis(0)
        dy = -self.gamepad.get_axis(1)
        dz = -self.gamepad.get_axis(self.z_axis)
        if abs(dx) < self.deadband:
            dx = 0.0
        if abs(dy) < self.deadband:
            dy = 0.0
        if abs(dz) < self.deadband:
            dz = 0.0
        A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
        B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
        X_pressed = self.gamepad.get_button(2)
        Y_pressed = self.gamepad.get_button(3)
        START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
        if A_pressed or B_pressed or START_pressed:
            self.lastpress = curr_time
        return [dx, dy, dz], A_pressed, B_pressed, START_pressed, X_pressed, Y_pressed


def sim_play(goal_numbers, iter):
    # filename = sys.argv[1]
    filename = "test" + str(goal_numbers+1) 
    tasks = str(goal_numbers+1)
    env_goals = int(goal_numbers+1)
    env = SimpleEnv(env_goals)
    
    quit = False
    target_goal = 0
    sigma_d = np.identity(3) * 0.001
    demos_savename = "demos/" + str(filename) + ".pkl"
    data_savename = "runs/" + str(filename) + ".pkl"
    cae_model = 'models/' + 'cae_' + str(tasks)
    class_model = 'models/' + 'class_' + str(tasks)
    model = Model(class_model, cae_model)
    interface = Joystick()

    print('[*] Initializing recording...')
    demonstration = []
    data = []
    effort = []
    final_error = []
    help = []
    velocity_arr = []
    record = False
    translation_mode = True
    start_time = time.time()
    scaling_trans = 0.1
    scaling_rot = 0.2
    assist = False
    assist_start = 3.
    steptime = 0.1
    goals = pickle.load(open("goals/goals" + str(env_goals) + ".pkl", "rb"))
    flag = 0

    print('[*] Main loop...')
    state = env.reset()
    print('[*] Waiting for start...')
    state = state[-1]
    start_q = state['joint_position']
    start_pose = state['ee_position']

    while not quit:
        # env.reset()
        state = env.state()
        s = state['joint_position'].tolist()

        u, start, mode, stop, X_in, Y_in = interface.input()
        if stop:
            # pickle.dump( demonstration, open( demos_savename, "wb" ) )
            # pickle.dump(data, open( data_savename, "wb" ) )
            # print(data[0])
            # pickle.dump(effort, open("effort/" + env_goals + "_1.pkl", "wb"))
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            quit = True
        
        if (start and not record) or flag == 0:
            record = True
            flag = 1
            start_time = time.time()
            assist_time = time.time()
            print('[*] Recording the demonstration...')
        
        # if mode or len(effort) == 450:
        if len(effort)>50:
            if np.mean(velocity_arr[len(velocity_arr)-10:len(velocity_arr)]) < 0.04 or len(help) > 300:
                record = False
                assist = False
                final_error.append(state["ee_position"] - goals[target_goal])
                env.reset()
                print("Number of points recorded = ", len(effort))
                print("Length of Help Array = ", len(help))
                pickle.dump(effort, open("effort/Effort/" + str(env_goals) + "/" + str(iter+1) + "_" + str(target_goal+1) + ".pkl", "wb"))
                pickle.dump(help, open("effort/Alpha/" + str(env_goals) +  "/" + str(iter+1) + "_" + str(target_goal+1) + ".pkl", "wb"))
                if target_goal<int(env_goals)-1:
                    target_goal += 1 
                    effort = []
                    help = []
                    print(len(effort))
                    print("Moving on to goal number", target_goal+1)
                else:
                    target_goal = target_goal
                    print(final_error)
                    pickle.dump(final_error, open("effort/Error/" + str(env_goals) + "/" + str(iter+1) + ".pkl", "wb"))
                    print("break executed")
                    p.disconnect()
                    break
                flag = 0
                time.sleep(3)
            

        xdot_h = np.zeros(6)
        xdot_h[:3] = scaling_trans * np.asarray(u)
        

        x_pos = state['ee_position']

        # xdot_h[:3] = (goals[target_goal] - x_pos) * 1 * flag
        xdot_h[:3] = np.random.multivariate_normal(goals[target_goal] - x_pos, sigma_d) * flag
        xdot_h[:3] = np.clip(xdot_h[:3], -0.1, 0.1)

        alpha = model.classify(start_pose.tolist() + x_pos.tolist())
        alpha = min(alpha,0.7)
        z = model.encoder(start_pose.tolist() + x_pos.tolist())
        a_robot = model.decoder(z, x_pos.tolist())
        xdot_r = np.zeros(6)
        xdot_r[:3] = a_robot
        
        curr_time = time.time()
        if record and curr_time - assist_time >= assist_start and not assist:    
            print("[*] Assistance started...")
            assist = True

        if assist:
            # alpha = 0
            xdot = alpha * 1 * xdot_r + (1 - alpha)* 2 * xdot_h 
        else:
            xdot = xdot_h

        if x_pos[2] < 0.1 and xdot[2] < 0:
            xdot[2] = 0  

        # qdot = xdot2qdot(xdot, state)
        # qdot = qdot.tolist()
        
        if record and curr_time - start_time >= steptime:
            s = state['joint_position'].tolist()
            demonstration.append(start_q + s)
            elapsed_time = curr_time - assist_time
            xdoth = xdot_h[:3]
            effort.append((1-alpha)*(abs(xdoth[0])+abs(xdoth[1])+abs(xdoth[2])))
            velocity_arr.append(np.linalg.norm(xdot[:3]))
            help.append(alpha)
            # qdot_h = xdot2qdot(xdot_h, state).tolist()
            # qdot_r = xdot2qdot(xdot_r, state).tolist()
            # data.append([elapsed_time] + [s] + [qdot_h] + [qdot_r] + [float(alpha)])
            start_time = curr_time
            # print(np.linalg.norm(goals[target_goal] - x_pos))
            # print(float(alpha))
            print(len(help))
            print("Xdot_h = {0:1.2f}, Xdot_r = {1:1.2f}, Xdot = {2:1.2f}, alpha = {3:1.2f}".format(np.linalg.norm(xdot_h[:3]), np.linalg.norm(xdot_r[:3]), np.linalg.norm(xdot[0:3]), alpha))

        env.step(0.1*xdot[:3])
        
        
def main():
    max_goals = sys.argv[1]
    for goal_numbers in range(24,int(max_goals)+1):
        for iter in range(5):
            sim_play(goal_numbers, iter)


if __name__ == "__main__":
    main()
