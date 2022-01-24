import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_cae import CAE
from train_classifier import Net
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU
torch.cuda.empty_cache()

class Model(object):

    def __init__(self, classifier_name, cae_name):
        self.class_net = Net()
        self.cae_net = CAE()
        
        model_dict = torch.load(classifier_name, map_location='cpu')
        self.class_net.load_state_dict(model_dict)
        
        # model_dict = torch.load(cae_name, map_location='cpu')
        # self.cae_net.load_state_dict(model_dict)

        self.class_net.eval
        self.cae_net.eval

    def classify(self, c):
        labels = self.class_net.classifier(torch.FloatTensor(c))
        confidence = F.softmax(labels, dim=0)
        return confidence.data[0].numpy()

    # def encoder(self, c):
    #     z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
    #     return z_mean_tensor.tolist()

    # def decoder(self, z, s):
    #     z_tensor = torch.FloatTensor(z + s)
    #     a_predicted = self.cae_net.decoder(z_tensor)
    #     return a_predicted.data.numpy()


def run(gstar, iter, vae):
    filename = sys.argv[1]
    tasks = sys.argv[2]
    demos_savename = "demos/" + str(filename) + ".pkl"
    data_savename = "runs/" + str(gstar) + "_" + vae + "_" + str(iter)+ ".pkl"
    cae_model = 'models/' + '0_cae_' + str(tasks)
    class_model = 'models/' + '0_class_' + str(tasks)
    model = Model(class_model, cae_model)
    # print('[*] Initializing recording...')
    demonstration = []
    data = []

    state = 0
    # goal = np.random.multivariate_normal(GOAL_D, SIGMA_D)
    a_h = np.random.normal(gstar-state, sigma_d)
    alpha = model.classify([state, a_h])
    alpha = min(alpha, 0.85)
    print(alpha)
    return alpha


def main():
    x = []
    g_range = np.arange(5,6,1)
    vae = "novae"
    for gx in g_range:
        final = []
        for iter in range(5):
            final_state = run(gx, iter, vae)
    #         poi = final_state[1]
    #         final.append(final_state)
    #         print("gx: {0:1.3f} iter: {1} xreal: {2:1.3f}".format(gx,iter,poi))
    #     x.append(np.mean(final, axis=0).tolist())
    # print([g_range, x])
    # pickle.dump([g_range, x], open("final_state.pkl", "wb"))
    # plt.plot(g_range.tolist(), x)
    # plt.show()

        

if __name__ == "__main__":
    main()
