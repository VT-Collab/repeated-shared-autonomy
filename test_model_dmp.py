import pygame
import sys
import os
import math
import numpy as np
import time
import pickle
from train_dmp import CAE
import torch
import copy

# Some constants
ALPHA = 0.1
BETA = 0.01
TAU = 1.0
T = 0.033 #time between data points (30fps => 1/30 sec)

class Model(object):

    def __init__(self, modelname):
        self.model = CAE()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval
        self.prev_y = torch.zeros(1,2)
        self.prev_y_dot = torch.zeros(1,2)
        # self.enable_dropout()

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def encoder(self, c):
        z = self.model.encoder(torch.FloatTensor(c))
        return z

    def action(self, z):
        y_ddot = torch.zeros(z.size())
        y_dot = torch.zeros(z.size())
        y = torch.zeros(z.size())

        y_ddot = 1 / TAU * ALPHA * (BETA * (z- self.prev_y) - self.prev_y_dot)
        y_dot = self.prev_y_dot + y_ddot * T
        y = self.prev_y + y_dot * T + 0.5 * y_ddot * T ** 2
        
        self.prev_y = y
        self.prev_y_dot = y_dot
        return y

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

    modelname = "models/dmp_1"

    clock = pygame.time.Clock()
    pygame.init()
    fps = 30

    joystick = Joystick()
    model = Model(modelname)

    world = pygame.display.set_mode([700,700])
    # position_player = np.random.random(2)
    # postition_blue = np.random.random(2)
    # postition_green = np.random.random(2)
    # postition_gray = np.random.random(2)
    position_player = [0.5,0.5]
    postition_blue = [0.1, 0.4]#np.random.random()]
    postition_green = [0.9, 0.56]#np.random.random()]
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
        
        # print("Z_mean: ",z_mean)
        # print("Z_std: ",z_std)
        actions = np.zeros((100, 2))
        for idx in range(100):
            z = model.encoder(c)
            a_robot = model.action(z)
            actions[idx,:] = a_robot.detach().numpy()
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
