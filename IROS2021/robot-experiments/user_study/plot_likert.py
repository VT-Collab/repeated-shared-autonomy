import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():


    labels = ['Recognize', 'Replicate', 'Improve', 'Deploy', 'Prefer']
    likert = [5.7, 5.5, 6.1, 5.65, 5.4]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, likert, width)

    ax.set_ylabel('User Rating')
    ax.set_title('Survey Results')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()