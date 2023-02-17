import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import os
import copy
# from utils import deform

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

class MotionData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.target = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = torch.Tensor(self.data[idx])
        label = torch.tensor(self.target[idx])
        label = label.type(torch.LongTensor)
        return (traj, label)


# GRU based classifier
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss(weight = torch.Tensor([20., 1.]))
        latent_dim = 10

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
            nn.Linear(5, 2)
        )

    def classify(self, x):
        o, h = self.gru(x)
        y = self.fcn(h)
        y = F.softmax(y.squeeze(), dim=0)
        return y

    def forward(self, x):
        c = x[0]
        y_true = x[1]
        o, h = self.gru(c)
        y_output = self.fcn(o)
        # print(y_output)
        shape = y_output.shape
        # y_output = torch.transpose(y_output, shape[2], shape[1])
        # y_output = y_output.view(shape[0], shape[2], shape[1])
        y_output = y_output.view(-1, 2)
        # print(y_output)
        # sys.exit()
        y_true = y_true.view(-1)
        # y_true = y_true.unsqueeze(2)
        loss = self.loss(y_output, y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)

def deform(xi, start, length, tau):
    length += np.random.choice(np.arange(50, length))
    xi1 = copy.deepcopy(np.asarray(xi))
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(np.dot(A.T, A))
    U = np.zeros(length)
    gamma = np.zeros((length, 6))
    for idx in range(6):
        U[0] = tau[idx]
        gamma[:,idx] = np.dot(R, U)
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end+1,:] += gamma[0:end-start+1,:]
    return xi1

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
        tau = np.random.uniform([-0.0004]*6, [0.0004]*6)
        deform_samples = 20
        for _ in range(deform_samples):
            start = np.random.choice(np.arange(20, 100))
            traj[:,6:] = deform(traj[:,6:], start, len(traj), tau)
            data = np.column_stack((traj, actions)).tolist()
            label = np.ones(len(traj)).tolist()
            # label = 1.
            dataset.append([data, label])

    pickle.dump(dataset, open(savename, "wb"))
    print("[*] I have this many trajectories: ", len(dataset))

    model = Net().to(device)
    dataname = 'data/' + 'class_' + "_".join(tasklist) + '.pkl'
    savename = 'models/' + 'class_' + "_".join(tasklist)

    EPOCH = 600
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))

    inputs = [element[0] for element in raw_data]
    targets = [element[1] for element in raw_data]
    # print(targets)
    train_data = MotionData(inputs, targets)
    BATCH_SIZE_TRAIN = 5
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
    required_tasks = [["push1"], ["push2"], ["cut1"], ["cut2"], ["scoop1"], ["scoop2"],\
                      ["open1"], ["open2"]]
    max_demos = 15
    for tasklist in required_tasks:
        print("[*] Training for task: ", tasklist)
        train_classifier(tasklist, max_demos)

if __name__ == "__main__":
    main()
