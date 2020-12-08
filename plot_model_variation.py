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
    position_gray = [0.5, 0.1]
    positions_blue = []
    positions_green = []
    obs_positions = []
    start_states = []
    for i in range(3):
        position_blue = [np.random.random(), np.random.random()]
        position_green = [np.random.random(), np.random.random()]
        positions_blue.append(position_blue)
        positions_green.append(position_green)
        obs_position = position_blue + position_green + position_gray
        start_state = obs_position + position_player
        obs_positions.append(obs_position)
        start_states.append(start_state)
    

    row = 0
    scale = 30
    heatmap_1 = np.zeros((scale, scale))
    heatmap_2 = np.zeros((scale, scale))
    heatmap_3 = np.zeros((scale, scale))

    for i in range(0,scale):
        col = 0
        for j in range(0,scale):
            x = i/float(scale)
            y = j/float(scale)
            q = np.asarray([x, y])
            
            s_1 = obs_positions[0] + q.tolist()
            c_1 = start_states[0] + q.tolist()
            s_2 = obs_positions[1] + q.tolist()
            c_2 = start_states[1] + q.tolist()
            s_3 = obs_positions[2] + q.tolist()
            c_3 = start_states[2] + q.tolist()
            
            z_mean_1, z_std_1 = model.encoder(c_1)
            z_mean_1 = z_mean_1[0]
            z_std_1 = z_std_1[0]
            z_mean_2, z_std_2 = model.encoder(c_2)
            z_mean_2 = z_mean_2[0]
            z_std_2 = z_std_2[0]
            z_mean_3, z_std_3 = model.encoder(c_3)
            z_mean_3 = z_mean_3[0]
            z_std_3 = z_std_3[0]

            actions_1 = np.zeros((100, 2))
            actions_2 = np.zeros((100, 2))
            actions_3 = np.zeros((100, 2))
            for idx in range(100):
                z_1 = z_mean_1 + np.random.normal() * z_std_1
                z_2 = z_mean_2 + np.random.normal() * z_std_2
                z_3 = z_mean_3 + np.random.normal() * z_std_3
                a_robot_1 = model.decoder([z_1], s_1)
                a_robot_2 = model.decoder([z_2], s_2)
                a_robot_3 = model.decoder([z_3], s_3)
                actions_1[idx,:] = a_robot_1
                actions_2[idx,:] = a_robot_2
                actions_3[idx,:] = a_robot_3
            # a_robot = np.mean(actions, axis=0)
            heatmap_1[row,col] = np.std(actions_1)
            heatmap_2[row,col] = np.std(actions_2)
            heatmap_3[row,col] = np.std(actions_3)
            # print(np.std(actions))
            col += 1
            # print(col)
        row += 1
    # print(heatmap)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    i = 0
    for ax in axs:
        position_green = positions_green[i]
        position_blue = positions_blue[i]
        ax.plot(position_blue[0]*scale, position_blue[1]*scale, 'bo', markersize=14)
        ax.plot(position_green[0]*scale, position_green[1]*scale, 'go', markersize=14)
        ax.plot(position_gray[0]*scale, position_gray[1]*scale, 'ko', markersize=14)
        ax.plot(position_player[0]*scale, position_player[1]*scale, 'mo', markersize=14) 
        i+= 1
    axs[0].imshow(heatmap_1.T, cmap='hot', interpolation='nearest')
    # axs[0].set_title('BETA = 0.001')
    axs[1].imshow(heatmap_2.T, cmap='hot', interpolation='nearest')
    # axs[1].set_title('BETA = 0.0001')
    axs[2].imshow(heatmap_3.T, cmap='hot', interpolation='nearest')
    # axs[2].set_title('BETA = 0.000001')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
