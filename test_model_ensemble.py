import pygame
import sys
import os
import math
import numpy as np
import time
import pickle
from train_model_variation import CAE
import torch
import copy


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

    name = "models/vae_ensemble_"

    clock = pygame.time.Clock()
    pygame.init()
    fps = 30

    joystick = Joystick()
    # model = Model(modelname)
    models = []
    for i in range(1,11):
        num = i+1
        modelname = name + str(num)
        print(modelname)
        model = Model(modelname)
        models.append(model)

    world = pygame.display.set_mode([700,700])
    # position_player = np.random.random(2)
    # postition_blue = np.random.random(2)
    # postition_green = np.random.random(2)
    # postition_gray = np.random.random(2)
    position_player = [0.5,0.5]
    postition_blue = [0.1, np.random.random()]
    postition_green = [0.9, np.random.random()]
    postition_gray = [0.5, 0.1]
    # obs_position = postition_blue.tolist() + postition_green.tolist() + postition_gray.tolist()
    obs_position = postition_blue + postition_green + postition_gray


    player = Player(position_player)
    blue = Object(postition_blue, [0, 0, 255])
    green = Object(postition_green, [0, 255, 0])
    gray = Object(postition_gray, [128, 128, 128])

    sprite_list = pygame.sprite.Group()
    sprite_list.add(player)
    sprite_list.add(blue)
    sprite_list.add(green)
    sprite_list.add(gray)

    world.fill((0,0,0))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)

    # start_state = obs_position + position_player.tolist()
    start_state = obs_position + position_player
    startt = time.time()

    while True:

        q = np.asarray([player.x, player.y])
        s = obs_position + q.tolist()
        c = start_state + q.tolist()
        # z_mean, z_std = model.encoder(c)
        # z_mean = z_mean[0]
        # z_std = z_std[0]
        # print("Z_mean: ",z_mean)
        # print("Z_std: ",z_std)
        actions = np.zeros((5, 2))
        for idx in range(1,11):
            model = models[idx]
            z_mean, z_std = model.encoder(c)
            z = z_mean[0] #+ np.random.normal() * z_std
            a_robot = model.decoder([z], s)
            actions[idx,:] = a_robot
        a_robot = np.mean(actions, axis=0)
        print("action std: ", np.std(actions))

        beta = 100/np.std(actions)

        action, start, stop = joystick.input()
        if stop:
            pygame.quit(); sys.exit()

        q += np.asarray(a_robot) * 0.003 + np.asarray(action) * 0.007

        # dynamics
        player.update(q)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    main()
