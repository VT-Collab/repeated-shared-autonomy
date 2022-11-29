import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def plot(data, axs):
    # plotting for debug
    pca = PCA(n_components=2)
    for i, output in enumerate(data["data"]):
        plt_data = pca.fit_transform(output)
        plt_data[:,0] = plt_data[:,0]/max(plt_data[:,0])
        plt_data[:,1] = plt_data[:,1]/max(plt_data[:,1])
        axs[i].clear()
        axs[i].set_xlim([-1.5, 1.5])
        axs[i].set_ylim([-1.5, 1.5])
        axs[i].plot(plt_data[np.where(data["gt"]==0),0],plt_data[np.where(data["gt"]==0),1], 'bx')
        axs[i].plot(plt_data[np.where(data["gt"]==1),0],plt_data[np.where(data["gt"]==1),1], 'rx')
    plt.draw()
    plt.pause(0.0001)

def main():
    plot_data = pickle.load(open("plot_data.pkl", "rb"))
    fig, axs = plt.subplots(int(len(plot_data[0]["data"])/2), 2)
    axs = axs.flat
    plt.ion()
    plt.show()
    for item in plot_data:
        if item["epoch"] % 10 == 0:
            plot(item, axs)
            plt.suptitle("Epoch: {} batch: {}".format(item["epoch"], item["batch"]))

if __name__ == "__main__":
    main()