import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from tasks import TASKSET

# tasklist = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
#                   ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
#                   ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"]]#,
# tasklist =      [["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
#                   ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
#                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]

tasklist = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
                  ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
                  ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
                  ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
                  ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
                  ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
                  ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
                  ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
                  ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                  ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]

rows = 4
fig, axs = plt.subplots(rows, int(len(tasklist)/rows))

colormap = plt.get_cmap('gist_rainbow')
colors = {}
for i, task in enumerate(TASKSET):
    colors[task] = colormap(1.*i/8)
for taskset, ax in zip(tasklist, axs.ravel()):
    model = "_".join(taskset)
    folder = "runs/alpha_0.2"
    latent_z = {}
    for filename in os.listdir(folder):
        if not filename[0] == "m":
            continue
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        if not traj["model"] == model:
            continue
        task = traj["task"]
        z = [item[-2] for item in traj["data"]]
        if task in latent_z:
            z = np.array(z).reshape(1, len(z), 2)
            latent_z[task] = np.append(latent_z[task], z, axis = 0)
        else:
            latent_z[task] = np.array(z).reshape(1, len(z), 2)
    for task in taskset:
        # print(taskset)
        z = np.mean(latent_z[task], axis=0)
        ax.plot(z[:,0], z[:,1], label=task, color=colors[task])
    # ax.legend()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(tasklist))

plt.suptitle("evolution of latent z over models")
plt.show()