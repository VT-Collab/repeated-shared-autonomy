import time
import numpy as np
import pickle
import pygame
import sys
import random
import os
import torch
import torch.nn as nn
from env import SimpleEnv
from train_cae import CAE
from train_classifier import Net
import torch.nn.functional as F
import pybullet as p


HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
SIGMA_D = np.identity(3) * 0.001

def plot_err():
    # get final state for each run from ours and ensemble per run
    # avg over runs and then over models for ours
    # plot


def main():
    plot_err()
    


if __name__ == '__main__':
    main()