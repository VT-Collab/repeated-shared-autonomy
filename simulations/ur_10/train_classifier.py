import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from sklearn.utils import shuffle
import sys
import os
import copy 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

# collect dataset
class MotionData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.target = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = torch.Tensor(self.data[idx])
        label = torch.tensor(self.target[idx])
        return (traj, label)


# conditional autoencoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.MSELoss()
        latent_dim = 5

        #RNN
        self.gru = nn.GRU(
            input_size = 18,
            hidden_size = latent_dim,
            num_layers = 1,
            batch_first = True
            )
        self.fcn = nn.Sequential(
            nn.Linear(latent_dim, 5),
            nn.Tanh(),
            nn.Linear(5, 1)
            # nn.LogSoftmax()
        )

    def classify(self, x):
        o, h = self.gru(x)
        # print("o:", o.shape)
        # print("h:", h.shape)
        y = self.fcn(h)
        # print("y:", y.shape)
        return y

    def forward(self, x):
        c = x[0]
        y_true = x[1]
        # print(c.shape)
        o, h = self.gru(c)
        y_output = self.fcn(o)
        y_true = y_true.unsqueeze(2)
        loss = self.loss(y_output, y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)


def deform(xi, start, length, tau):
    xi1 = copy.deepcopy(np.asarray(xi))
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(np.dot(A.T, A))
    U = np.zeros(length)
    gamma = np.zeros((length, len(tau)))
    for idx in range(len(tau)):
        U[0] = tau[idx]
        gamma[:,idx] = np.dot(R, U)
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end,:] += gamma[0:end-start,:]
    return xi1

# train cAE
def train_classifier(tasklist, max_demos):

    tasks = []
    for task in tasklist:
        for i in range(1, max_demos+1):
            tasks.append(task + "_" + str(i) + ".pkl")

    dataset = []
    folder = 'demos'

    savename = 'data/' + 'class_' + "_".join(tasklist) + '.pkl'
    for filename in os.listdir(folder):
        if not filename in tasks:
            continue
        demo = pickle.load(open(folder + "/" + filename, "rb"))

        # Human's demonstrations
        data = [item[0] + item[1] for item in demo]
        # traj = torch.Tensor(traj).unsqueeze(0)
        label = np.zeros(len(demo)).tolist()
        # label = 0.
        dataset.append([data, label])

        # Deformations
        traj = np.array([item[0] for item in demo])
        actions = np.array([item[1] for item in demo])
        tau = np.random.uniform([-0.07]*6, [0.07]*6)
        traj[:,6:] = deform(traj[:,6:], 0, len(traj), tau)
        data = np.column_stack((traj, actions)).tolist()
        label = np.ones(len(traj)).tolist()
        # label = 1.
        dataset.append([data, label])

    pickle.dump(dataset, open(savename, "wb"))
    print("[*] I have this many trajectories: ", len(dataset))

    model = Net().to(device)
    dataname = 'data/' + 'class_' + "_".join(tasklist) + '.pkl'
    savename = 'models/' + 'class_' + "_".join(tasklist)

    EPOCH = 500
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))

    inputs = [element[0] for element in raw_data]
    targets = [element[1] for element in raw_data]
    # print(targets)
    train_data = MotionData(inputs, targets)
    BATCH_SIZE_TRAIN = 3
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)

def main():
    max_demos = 5
    tasklist = ["push1"]
    train_classifier(tasklist, max_demos)

if __name__ == "__main__":
    main()
