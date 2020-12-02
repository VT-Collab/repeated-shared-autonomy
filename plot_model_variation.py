import pygame
import sys
import os
import math
import numpy as np
import time
import pickle
from train_model_variation import CAE
import torch
import matplotlib.pyplot as plt

class Model(object):

    def __init__(self, modelname):
        self.model = CAE()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval
        self.enable_dropout()

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def encoder(self, c):
        z_mean, z_log_var = self.model.encoder(torch.FloatTensor(c))
        return z_mean.tolist(), torch.exp(0.5*z_log_var).tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.model.decoder(z_tensor)
        return a_predicted.data.numpy()

class Joystick(object):

    def __init__(self):
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        if abs(z1) < self.DEADBAND:
            z1 = 0.0
        z2 = self.gamepad.get_axis(1)
        if abs(z2) < self.DEADBAND:
            z2 = 0.0
        start = self.gamepad.get_button(1)
        stop = self.gamepad.get_button(0)
        return [z1, z2], start, stop


class Object(pygame.sprite.Sprite):

    def __init__(self, position, color):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25,25))
        self.image.fill(color)
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = (self.x * 500) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 500) + 100 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self, position):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50,50))
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = (self.x * 500) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 500) + 100 - self.rect.size[1] / 2

    def update(self, s):
        self.rect = self.image.get_rect(center=self.rect.center)
        self.x = s[0]
        self.y = s[1]
        self.rect.x = (self.x * 500) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 500) + 100 - self.rect.size[1] / 2


def main():

    modelname = "models/vae_model_b_0001"
    model = Model(modelname)
    position_player = [0.5,0.5]
    position_blue = [0.5, 0.2]#np.random.random()]
    position_green = [0.5, 0.8]#np.random.random()]
    position_gray = [0.1, 0.5]
    obs_position = position_blue + position_green + position_gray
    start_state = obs_position + position_player

    row = 0
    N = 50
    heatmap = np.zeros((N, N))
    for i in range(0,100, 2):
        col = 0
        for j in range(0,100, 2):
            x = i/100.0
            y = j/100.0
            q = np.asarray([x, y])
            s = obs_position + q.tolist()
            c = start_state + q.tolist()
            z_mean, z_std = model.encoder(c)
            z_mean = z_mean[0]
            z_std = z_std[0]
            # print("Z_mean: ",z_mean)
            # print("Z_std: ",z_std)
            actions = np.zeros((100, 2))
            for idx in range(100):
                z = z_mean + np.random.normal() * z_std
                a_robot = model.decoder([z], s)
                actions[idx,:] = a_robot
            a_robot = np.mean(actions, axis=0)
            heatmap[row,col] = np.std(actions)
            # print(np.std(actions))
            col += 1
            # print(col)
        row += 1
    # print(heatmap)
    fig = plt.figure()
    plt.plot(position_blue[0]*N, position_blue[1]*N, 'bo')
    plt.plot(position_green[0]*N, position_green[1]*N, 'go')
    plt.plot(position_gray[0]*N, position_gray[1]*N, 'ko')
    plt.plot(position_player[0]*N, position_player[1]*N, 'mo') 
    plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main()
