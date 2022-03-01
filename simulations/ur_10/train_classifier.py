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
        # self.data, self.target = shuffle(self.data, self.target)
        self.target = torch.as_tensor(self.target).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        state = torch.FloatTensor(item[1]).to(device)
        label = self.target[idx]
        return (snippet, state, label)


# conditional autoencoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()
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
            nn.Linear(5, 2)
        )

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        c = x[0]
        s = x[1]
        h, _ = self.gru(c)
        y_output = self.fcn(h)
        y_true = x[2]
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
def train_classifier(tasks):

    dataset = []
    folder = 'demos'

    true_cnt = 0
    false_cnt = 0

    savename = 'data/' + 'class_' + str(tasks) + '.pkl'
    for filename in os.listdir(folder):
        if not filename in tasks:
            continue
        demo = pickle.load(open(folder + "/" + filename, "rb"))
        traj = [item[0] for item in demo]
        action = [item[1] for item in demo]
        n_states = len(traj)
        for idx in range(n_states):
            home_state = traj[idx][:6]
            position = traj[idx][6:]
            traj_type = 0
            dataset.append((home_state + position + action[idx], traj_type))
            true_cnt += 1

        snippets = np.array_split(traj, 1)
        deformed_samples = 2
        for snip in snippets:
                tau = np.random.uniform([-0.07]*6, [0.07]*6)
                deform_len = len(snip)
                start = 0
                for i in range(deformed_samples):
                    snip_deformed = deform(snip[:,6:], 0, deform_len, tau)
                    snip[:,6:] = snip_deformed
                    n_states = len(snip)
                    for idx in range(start, deform_len):
                        home_state = snip[idx][:6].tolist()
                        position = snip[idx][6:] 
                        traj_type = 1
                        dataset.append((home_state + position.tolist() + action[idx], traj_type))
                        false_cnt += 1

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[-1])
    print("[*] I have this many subtrajectories: ", len(dataset))
    print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))

    model = Net().to(device)
    dataname = 'data/' + 'class_' + str(tasks) + '.pkl'
    savename = 'models/' + 'class_' + str(tasks)

    EPOCH = 500
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    raw_data = random.sample(raw_data, len(raw_data))

    inputs = [element[:4] for element in raw_data]
    targets = [element[4] for element in raw_data]
    # print(inputs[0])

    train_data = MotionData(inputs, targets)
    BATCH_SIZE_TRAIN = int(train_data.__len__() / 10.)
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
    keys = ["push1"]
    tasklist = []
    for key in keys:
        for i in range(1, demo_num+1):
            tasklist.append(key + "_" + str(i) + ".pkl")
    train_classifier(tasklist)

if __name__ == "__main__":
    main()
