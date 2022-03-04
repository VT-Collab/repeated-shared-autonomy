import matplotlib.pyplot as plt
import pickle
import numpy as np


def main():

    tasknames = ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2",\
                      "open1", "open2"]
    """steptime = 0.4 data"""
    data_dict = pickle.load(open("confidences_steptime_0.1.pkl", "rb"))
    data_04 = []
    for key in tasknames:
        data_04.append(data_dict[key])
    """steptime = 0.1 data"""
    data_dict = pickle.load(open("confidences_demos_15.pkl", "rb"))
    data_01 = []
    for key in tasknames:
        data_01.append(data_dict[key])

    fig, axs = plt.subplots(1, 2, figsize=(10,20))
    data_04 = np.around(np.array(data_04), decimals=1)
    data_01 = np.around(np.array(data_01), decimals=1)

    axs[0].imshow(data_04)
    axs[1].imshow(data_01)
    axs[0].set_title("5 Human demos")
    axs[1].set_title("15 Human demos")
    for ax in axs:
        ax.set_xticks(np.arange(len(tasknames)))
        ax.set_xticklabels(tasknames)
        ax.set_yticks(np.arange(len(tasknames)))
        ax.set_yticklabels(tasknames)

    for i in range(len(tasknames)):
        for j in range(len(tasknames)):
            text = axs[0].text(j, i, data_04[i, j], ha="center", va="center", color="w")
            text = axs[1].text(j, i, data_01[i, j], ha="center", va="center", color="w")

    plt.show()
if __name__ == "__main__":
    main()