import pygame
import sys
import os
import math
import numpy as np
import time
import pickle
import torch
import copy
from train_classifier import Net
from train_model import CAE
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Classifer
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
        # self.enable_dropout()

    def classify(self, c):
        labels = self.class_net.classifier(torch.FloatTensor(c))
        confidence = F.softmax(labels, dim=0)
        return confidence.data[0].numpy()

    def encoder(self, c):
        z_tensor = self.cae_net.encoder(torch.FloatTensor(c))
        return z_tensor.tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.cae_net.decoder(z_tensor)
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

# train cAE
def main():

    classifier_name = "models/classifier_dist"
    cae_name = "models/cae_model"

    model = Model(classifier_name, cae_name)

    clock = pygame.time.Clock()
    pygame.init()
    fps = 30

    joystick = Joystick()


    world = pygame.display.set_mode([700,700])
    position_player = np.random.random(2)
    postition_blue = np.random.random(2)
    postition_green = np.random.random(2)
    postition_gray = np.random.random(2)
    # position_player = [0.5,0.5]
    # postition_blue = [0.1, 0.4]#np.random.random()]
    # postition_green = [0.9, 0.56]#np.random.random()]
    # postition_gray = [0.5, 0.1]
    obs_position = postition_blue.tolist() + postition_green.tolist() + postition_gray.tolist()
    # obs_position = postition_blue + postition_green + postition_gray


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

    start_state = obs_position + position_player.tolist()
    # start_state = obs_position + position_player
    startt = time.time()

    while True:

        q = np.asarray([player.x, player.y])
        s = obs_position + q.tolist()
        c = start_state + q.tolist()
        actions = np.zeros((100, 2))
        zs = []

        confidence = 0.75 * model.classify(c)
        print(confidence)

        for idx in range(100):
            z = model.encoder(c)
            a_robot = model.decoder(z, s)
            zs += z
            actions[idx,:] = a_robot
        a_robot = np.mean(actions, axis=0)
        a_var = np.std(actions, axis=0)

        action, start, stop = joystick.input()
        if stop:
            pygame.quit(); sys.exit()

        q += 0.01 * (np.asarray(a_robot) * confidence + np.asarray(action) * (1 - confidence))

        # dynamics
        player.update(q)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)

if __name__ == "__main__":
    main()