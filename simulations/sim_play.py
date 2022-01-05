import time
import numpy as np
import pickle
import pygame
import sys
import random

import torch
import torch.nn as nn
from env import SimpleEnv
from train_cae import CAE
from train_classifier import Net
import torch.nn.functional as F

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

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        dx = self.gamepad.get_axis(0)
        dy = -self.gamepad.get_axis(1)
        dz = -self.gamepad.get_axis(4)
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


def main():
    env = SimpleEnv()
    quit = False

    filename = sys.argv[1]
    tasks = sys.argv[2]
    demos_savename = "demos/" + str(filename) + ".pkl"
    data_savename = "runs/" + str(filename) + ".pkl"
    cae_model = 'models/' + 'cae_' + str(tasks)
    class_model = 'models/' + 'class_' + str(tasks)
    model = Model(class_model, cae_model)
    interface = Joystick()

    print('[*] Initializing recording...')
    demonstration = []
    data = []
    record = False
    translation_mode = True
    start_time = time.time()
    scaling_trans = 0.1
    scaling_rot = 0.2
    assist = False
    assist_start = 3.
    steptime = 0.1
    traj1 = pickle.load(open("demos/1_1.pkl", "rb"))
    traj2 = pickle.load(open("demos/2_1.pkl", "rb"))
    start1 = traj1[15]
    start2 = traj2[15]
    # print(start1)


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
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            quit = True
        
        if start and not record:
            record = True
            start_time = time.time()
            assist_time = time.time()
            print('[*] Recording the demonstration...')
        
        if mode:
            env.reset()
            record = False
            assist = False

        if X_in:
            print("XINPUT")
            # env.step(start1)
            pos = [1.45969408e-03, -1.19984115e+00,  1.44616416e-03, -1.77202367e+00,  -6.51033560e-04,  1.35692320e+00,  7.87672825e-01]
            env.reset(pos)

            print(start1)
            print(state['joint_position'])
            print(state['ee_position'])

        
        if Y_in:
            print("YINPUT")
            # env.step(start2)
            pos = [0.03842222, -1.16415388, -0.12302732, -1.77853889, -0.07355825,  1.40489792,  0.68592709]
            env.reset(pos)
            print(start2)
            print(state['joint_position'])
            print(state['ee_position'])

        xdot_h = np.zeros(6)
        xdot_h[:3] = scaling_trans * np.asarray(u)

        x_pos = state['ee_position']

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
            xdot = alpha * 1 * xdot_r + (1 - alpha)*2 * xdot_h 
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
            # qdot_h = xdot2qdot(xdot_h, state).tolist()
            # qdot_r = xdot2qdot(xdot_r, state).tolist()
            # data.append([elapsed_time] + [s] + [qdot_h] + [qdot_r] + [float(alpha)])
            start_time = curr_time
            print(float(alpha))

        env.step(xdot[:3])
        

if __name__ == "__main__":
    main()
