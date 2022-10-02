import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def main():
    mpl.rcParams['legend.fontsize'] = 10

    fig, axs = plt.subplots(2,1,subplot_kw=dict(projection='3d'))
    # axs[0] = fig.gca(projection='3d')
    # axs[1] = fig.gca(projection='3d')

    folders = ['demos/pour', 'demos/place', 'demos/stir']
    deformed_trajs = pickle.load(open("deformed_trajs.pkl", "rb"))
    for folder in folders:
        for filename in os.listdir(folder):
            demo = pickle.load(open(folder + "/" + filename, "rb"))
            traj = [item["curr_pos"] for item in demo]
            # action = [item[1] for item in demo]
            traj = np.asarray(traj)
            axs[0].plot(traj[:,0], traj[:,1], traj[:,2], 'b', label='Demos')
            axs[1].plot(traj[:,3], traj[:,4], traj[:,5], 'b', label='Demos')
        for traj in deformed_trajs:
            # print(traj[:,:3])
            axs[0].plot(traj[:,0], traj[:,1], traj[:,2], 'r', label='Deforms')
            axs[1].plot(traj[:,3], traj[:,4], traj[:,5], 'r', label='Deforms')
    # ax.legend()
    
    plt.show()




if __name__ == '__main__':
    main()