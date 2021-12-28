import pickle
import copy
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def joint2pose(q):
    def RotX(q):
        return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
    def RotZ(q):
        return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def TransX(q, x, y, z):
        return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
    def TransZ(q, x, y, z):
        return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    H1 = TransZ(q[0], 0, 0, 0.333)
    H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
    H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
    H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
    H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
    H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
    H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
    H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
    H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
    return H[:,3][:3]

def blendColors(beta):
    gray = np.asarray([102./255, 102./255, 102./255])
    orange = np.asarray([255./255, 153./255, 0])
    purple = np.asarray([141./255, 95./255, 211./255])
    new_color = np.zeros(3)

    for i in range(3):
        new_color[i] = np.sqrt((1 - beta) * gray[i]**2 + beta * orange[i]**2)

    return new_color


def main():
    dataset = []
    data_loc = 'user/user4/runs'

    cup_ours_6 = []
    shelf_ours_6 = []
    cup_ours_3 = []
    cup_none = []
    shelf_none = []
    
    filename = 'figure_no_assist.pkl'
    traj = pickle.load(open(data_loc + "/" + filename, "rb"))
    n_states = len(traj)
    fig, ax = plt.subplots()


    gray = np.asarray([102./255, 102./255, 102./255])
    orange = np.asarray([255./255, 153./255, 0])
    purple = np.asarray([141./255, 95./255, 211./255])

    min_x = 0
    max_x = .5
    for idx in range(n_states):
        time = traj[idx][0]
        pose = joint2pose(traj[idx][1])
        if time <= 2.0:
            beta = 0
        else: 
            beta = traj[idx][-1] / (max_x - min_x)
        # if beta > max_x:
        #     max_x = beta
        # print(beta)
        # if beta > 1:
        #     beta = 1.
        color = blendColors(beta)
        plt.plot(-pose[0], pose[2], 'o', color=blendColors(beta))
        plt.ylim(-1, 1.5)
        plt.xlim(-1.5, 0)

    plt.show()


if __name__ == "__main__":
    main()