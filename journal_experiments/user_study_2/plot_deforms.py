import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def main():
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    folder = 'demos'
    deformed_trajs = pickle.load(open("deformed_trajs.pkl", "rb"))
    for filename in os.listdir(folder):
        demo = pickle.load(open(folder + "/" + filename, "rb"))
        traj = [item[0] for item in demo]
        action = [item[1] for item in demo]
        traj = np.asarray(traj)
        ax.plot(traj[:,6], traj[:,7], traj[:,8], 'b', label='parametric curve')
    # ax.legend()
    for traj in deformed_trajs:
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 'r', label='parametric curve')
    plt.show()




if __name__ == '__main__':
    main()