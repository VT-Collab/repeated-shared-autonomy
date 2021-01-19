import pygame
import sys
import os
import math
import numpy as np
import time
import pickle
from train_classifier import Net
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Classifer
class Model(object):

    def __init__(self, modelname):
        self.model = Net()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval
        # self.enable_dropout()

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def classify(self, x):
        labels = self.model.classifier(x)
        return F.softmax(labels, dim=0)

    def forward(self, x):
        c = torch.FloatTensor(x[0])
        s = x[1]
        c_output = self.classify(c)
        c_true = x[4]
        loss = self.loss(c_output, c_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)

def main():

    modelname = "models/classifier_dist"
    model = Model(modelname)
    
    positions_blue = []
    positions_green = []
    obs_positions = []
    start_states = []
    positions_start = []
    positions_gray = []
    for i in range(3):
        position_blue = [np.random.random(), np.random.random()]
        position_green = [np.random.random(), np.random.random()]
        position_player = [np.random.random(), np.random.random()]
        position_gray = [np.random.random(), np.random.random()]
        
        positions_blue.append(position_blue)
        positions_green.append(position_green)
        positions_start.append(position_player)
        positions_gray.append(position_gray)

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
            
            z_1 = model.classify(torch.FloatTensor(c_1))
            z_2 = model.classify(torch.FloatTensor(c_2))
            z_3 = model.classify(torch.FloatTensor(c_3))

            # label_1, _ = torch.max(z_1.data, 0)
            # label_2, _ = torch.max(z_2.data, 0)
            # label_3, _ = torch.max(z_3.data, 0)

            label_1 = z_1.data
            label_2 = z_2.data
            label_3 = z_3.data

            heatmap_1[row,col] = label_1[0]
            heatmap_2[row,col] = label_2[0]
            heatmap_3[row,col] = label_3[0]

            col += 1

        row += 1

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    i = 0
    for ax in axs:
        position_green = positions_green[i]
        position_blue = positions_blue[i]
        position_player = positions_start[i]
        position_gray = positions_gray[i]
        ax.plot(position_blue[0]*scale, position_blue[1]*scale,'bo',\
                        label="goal1", markersize=14)
        ax.plot(position_green[0]*scale, position_green[1]*scale,'go',\
                        label="goal2", markersize=14)
        ax.plot(position_gray[0]*scale, position_gray[1]*scale,'co',\
                        label="dummy", markersize=14)
        ax.plot(position_player[0]*scale, position_player[1]*scale, 'mo',\
                        label="start position", markersize=14) 
        i+= 1
    map1 = axs[0].imshow(heatmap_1.T, cmap='hot', interpolation='nearest')
    fig.colorbar(map1, ax=axs[0])
    map2 = axs[1].imshow(heatmap_2.T, cmap='hot', interpolation='nearest')
    fig.colorbar(map2, ax=axs[1])
    map3 = axs[2].imshow(heatmap_3.T, cmap='hot', interpolation='nearest')
    fig.colorbar(map3, ax=axs[2])
    
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.suptitle("True Contexts (Lighter means higher confidence)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
