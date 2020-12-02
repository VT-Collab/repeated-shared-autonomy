import pygame
import sys
import os
import math
import numpy as np
import time
import pickle
from train_model_variation import CAE as CAE_1
from train_model import CAE as CAE_2
import torch
import matplotlib.pyplot as plt

class Model_VAE(object):

    def __init__(self, modelname):
        self.model = CAE_1()
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

class Model_CAE(object):

    def __init__(self, modelname):
        self.model = CAE_2()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval
        self.enable_dropout()

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    def encoder(self, c):
        z_tensor = self.model.encoder(torch.FloatTensor(c))
        return z_tensor.tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.model.decoder(z_tensor)
        return a_predicted.data.numpy()


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
    model_1 = Model_VAE(modelname)
    model_2 = Model_VAE("models/vae_model_data_50")

    position_player = [0.5,0.5]
    position_blue = [0.2, np.random.random()]
    position_green = [0.8, np.random.random()]
    position_gray = [0.5, 0.1]
    obs_position = position_blue + position_green + position_gray
    start_state = obs_position + position_player

    row = 0
    N = 30
    heatmap_1 = np.zeros((N, N))
    heatmap_2 = np.zeros((N, N))
    heatmap_3 = np.zeros((N, N))

    for i in range(0,N):
        col = 0
        for j in range(0,N):
            x = i/float(N)
            y = j/float(N)
            q = np.asarray([x, y])
            s = obs_position + q.tolist()
            c = start_state + q.tolist()
            
            z_mean_1, z_std_1 = model_1.encoder(c)
            z_mean_1 = z_mean_1[0]
            z_std_1 = z_std_1[0]
            z_mean_2, z_std_2 = model_2.encoder(c)
            z_mean_2 = z_mean_2[0]
            z_std_2 = z_std_2[0]

            actions_1 = np.zeros((100, 2))
            actions_2 = np.zeros((100, 2))
            
            for idx in range(100):
                z_1 = z_mean_1 + np.random.normal() * z_std_1
                z_2 = z_mean_2 + np.random.normal() * z_std_2

                a_robot_1 = model_1.decoder([z_1], s)
                a_robot_2 = model_2.decoder([z_2], s)
                
                actions_1[idx,:] = a_robot_1
                actions_2[idx,:] = a_robot_2
                
            # a_robot = np.mean(actions, axis=0)
            heatmap_1[row,col] = np.std(actions_1)
            heatmap_2[row,col] = np.std(actions_2)
            
            # print(np.std(actions))
            col += 1
            # print(col)
        row += 1
    # print(heatmap)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax in axs:
        ax.plot(position_blue[0]*N, position_blue[1]*N, 'bo', markersize=14)
        ax.plot(position_green[0]*N, position_green[1]*N, 'go', markersize=14)
        ax.plot(position_gray[0]*N, position_gray[1]*N, 'ko', markersize=14)
        ax.plot(position_player[0]*N, position_player[1]*N, 'mo', markersize=14) 
    axs[0].imshow(heatmap_1.T, cmap='hot', interpolation='nearest')
    axs[0].set_title('100 Samples')
    axs[1].imshow(heatmap_2.T, cmap='hot', interpolation='nearest')
    axs[1].set_title('50 Samples')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
