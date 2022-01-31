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
from record_demos import GOAL, SIGMA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU
torch.cuda.empty_cache()

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


def run(gstar, iter, vae):
    cae_model = 'models/' + 'cae_' + str(GOAL)
    class_model = 'models/' + 'class_' + str(GOAL)
    model = Model(class_model, cae_model)
    demonstration = []
    data = []

    state = 0
    sigma_h = SIGMA
    sigma_d = SIGMA
    a_h = np.random.normal(gstar-state, sigma_h)
    if vae == "vae":
        z = model.encoder([state, 1.0])
        a_r = model.decoder(z, [state, 1.0])
    else:
        a_r = np.random.normal(GOAL-state, sigma_d)
    alpha = model.classify([state, a_h])
    alpha = min(alpha, 0.85)
    # print("a_h:{} alpha:{}".format(a_h, alpha))
    return [state + (1 - alpha) * a_h + alpha * a_r, alpha]


def main():
    x = []
    g_range = np.arange(0,GOAL*2+SIGMA/10.,SIGMA/10.)
    vae = "vae"
    max_iters = 10000
    avg_state = []
    for gx in g_range:
        final_state = []
        for iter in range(max_iters):
            res = run(gx, iter, vae)
            final_state.append(res[0])
        avg_state.append(np.mean(final_state))
    err = g_range - avg_state
    err = err.tolist()
    data = {}
    data["description"] = "1D error and gstar"
    data["avg_state"] = avg_state
    data["error"] = err
    data["vae"] = vae
    data["max_iters"] = max_iters
    pickle.dump(data, open("err_vs_gstar_" + vae + "_" + str(GOAL) + ".pkl", "wb"))
    plt.plot(g_range.tolist(), err)
    plt.show()     

if __name__ == "__main__":
    main()
