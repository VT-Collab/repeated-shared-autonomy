import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import sqrt

def main():

    means = [5.7, 5.5, 6.1, 5.65, 5.4]
    sem = [1.337/sqrt(10), 1.054/sqrt(10), 0.774/sqrt(10), 1.225/sqrt(10), 1.125/sqrt(10)]

    labels = ['Recognize', 'Replicate', 'Improve', 'Deploy', 'Prefer']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, means, width, yerr=sem)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

if __name__ == "__main__":
    main()