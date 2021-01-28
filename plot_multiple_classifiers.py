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

    def __init__(self, modelnames):
        self.models = []
        file_loc = "models/"

        for modelname in modelnames:
            model = Net()
            modelname = file_loc + modelname
            model_dict = torch.load(modelname, map_location='cpu')
            model.load_state_dict(model_dict)
            model.eval
            self.models.append(model)

    def classify(self, x):
        labels = []
        for model in self.models:
            output = model.classifier(x)
            label = F.softmax(output, dim=0)
            labels.append(label[0])
        return labels

def main():

    # Models we will be comparing
    modelnames = ["classifier_low_sigma", "classifier_med_sigma", "classifier_high_sigma"]
    model = Model(modelnames)
    
    # Generate random start and goal positions
    position_blue = [np.random.random(), np.random.random()]
    position_green = [np.random.random(), np.random.random()]
    position_player = [np.random.random(), np.random.random()]
    position_gray = [np.random.random(), np.random.random()]
    
    obs_position = position_blue + position_green + position_gray
    start_state = obs_position + position_player

    # Generate heatmap
    row = 0
    scale = 30
    heatmap_1 = np.zeros((scale, scale))
    heatmap_2 = np.zeros((scale, scale))
    heatmap_3 = np.zeros((scale, scale))

    # Iterate over all possible current positions and find whether true context
    for i in range(0,scale):
        col = 0
        for j in range(0,scale):
            x = i/float(scale)
            y = j/float(scale)
            q = np.asarray([x, y])
            
            state = obs_position + q.tolist()
            context = start_state + q.tolist()
            
            z = model.classify(torch.FloatTensor(context))

            label_1 = z[0].data
            label_2 = z[1].data
            label_3 = z[2].data

            heatmap_1[row,col] = label_1
            heatmap_2[row,col] = label_2
            heatmap_3[row,col] = label_3

            col += 1
        row += 1

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    i = 0
    for ax in axs:
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
    axs[0].set_title("Low variance (0.05)")
    map2 = axs[1].imshow(heatmap_2.T, cmap='hot', interpolation='nearest')
    fig.colorbar(map2, ax=axs[1])
    axs[1].set_title("Med variance (0.2)")
    map3 = axs[2].imshow(heatmap_3.T, cmap='hot', interpolation='nearest')
    fig.colorbar(map3, ax=axs[2])
    axs[2].set_title("High variance (0.5)")

    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.suptitle("Context Classification(Lighter means higher confidence)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
