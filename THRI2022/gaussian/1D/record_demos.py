import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random

GOAL = .5
SIGMA = .1
def generate_demo(filename):
        savename = "demos/" + str(filename) + ".pkl"
        state = np.zeros(filename)
        action = np.random.normal(GOAL-state, SIGMA, filename)
        data = np.column_stack((state, action))
        pickle.dump( data.tolist(), open( savename, "wb" ) )
        print(data.tolist())

def main():    
    generate_demo(250)


if __name__ == "__main__":
    main()