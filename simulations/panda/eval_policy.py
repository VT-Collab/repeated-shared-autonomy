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

def eval_policy(tasks, model_num, max_runs):
    cae_model = "models/cae_" + str(tasks) + "_" + str(model_num)
    class_model = "models/class_" + str(tasks) + "_" + str(model_num)
    env = SimpleEnv(tasks)
    for run in max_runs:





def main():
    


if __name__ == '__main__':
    main()