#!/usr/bin/env python
import rospy
import actionlib
import sys
import time
import numpy as np
import pygame
import pickle
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import copy, os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_classifier import Net
from train_cae import CAE
from waypoints import HOME

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100



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
        labels = self.class_net.classify(torch.FloatTensor(c))
        # print(labels)
        confidence = F.softmax(labels[0], dim=0)
        return confidence.data[0].numpy()
        # return labels.detach().numpy()

    def encoder(self, c):
        z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
        return z_mean_tensor.tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.cae_net.decoder(z_tensor)
        return a_predicted.data.numpy()

# GRU model
# class Model(object):

#     def __init__(self, classifier_name):#, cae_name):
#         self.class_net = Net()
#         # self.cae_net = CAE()
        
#         model_dict = torch.load(classifier_name, map_location='cpu')
#         self.class_net.load_state_dict(model_dict)
        
#         # model_dict = torch.load(cae_name, map_location='cpu')
#         # self.cae_net.load_state_dict(model_dict)

#         self.class_net.eval
#         # self.cae_net.eval

#     def classify(self, c):
#         label = self.class_net.classify(c)
#         return label.data.numpy()

    # def encoder(self, c):
    #     z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
    #     return z_mean_tensor.tolist()

    # def decoder(self, z, s):
    #     z_tensor = torch.FloatTensor(z + s)
    #     a_predicted = self.cae_net.decoder(z_tensor)
    #     return a_predicted.data.numpy()

# def deform(xi, start, length, tau):
#     xi1 = copy.deepcopy(np.asarray(xi))
#     A = np.zeros((length+2, length))
#     for idx in range(length):
#         A[idx, idx] = 1
#         A[idx+1,idx] = -2
#         A[idx+2,idx] = 1
#     R = np.linalg.inv(np.dot(A.T, A))
#     U = np.zeros(length)
#     gamma = np.zeros((length, 6))
#     for idx in range(6):
#         U[0] = tau[idx]
#         gamma[:,idx] = np.dot(R, U)
#     end = min([start+length, xi1.shape[0]-1])
#     xi1[start:end,:] += gamma[0:end-start,:]
#     return xi1

