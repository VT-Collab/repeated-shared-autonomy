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

        # Encoder
        self.classifier = nn.Sequential(
            nn.Linear(6, 15),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(15, 20),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(20, 7),
            nn.Tanh(),
            nn.Linear(7, 2)
        )

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        c = x[0]
        s = x[1]
        y_output = self.classify(c)
        y_true = x[2]
        loss = self.loss(y_output, y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)


def deform(xi, start, length, tau):
    length += np.random.choice(np.arange(int(length/10), length))
    xi1 = copy.deepcopy(np.asarray(xi))
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(np.dot(A.T, A))
    U = np.zeros(length)
    gamma = np.zeros((length, 3))
    for idx in range(3):
        U[0] = tau[idx]
        gamma[:,idx] = np.dot(R, U)
    end = min([start+length, xi1.shape[0]-1])
    print(start)
    print(end)
    xi1[start:end+1,:] += gamma[0:end-start+1,:]
    return xi1

# train cAE
def train_classifier(tasks, model_no):

    # tasks = int(sys.argv[1])
    tasks = int(tasks)

    dataset = []
    folder = 'demos/Noisy_Demos'
    lookahead = 0
    noiselevel = 0.05
    deformed_trajs = []
    # noisesamples = 3

    true_cnt = 0
    false_cnt = 0

    savename = 'data/' + 'class_' + str(tasks) + "_" + str(model_no) +'.pkl'
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        n_states = len(traj)
        home_state = traj[0]
        for idx in range(n_states):
            position = traj[idx]
            traj_type = 0 # 0 means real trajectory
            dataset.append((home_state.tolist() + position.tolist(), position.tolist(), traj_type))
            true_cnt += 1

        snippets = np.array_split(traj, 1)
        deformed_samples = 2
        for snip in snippets:
            tau = np.random.uniform([-0.00045]*3, [0.00045]*3)
            deform_len = len(snip)
            start = np.random.choice(np.arange(0, int(deform_len/2)))
            for i in range(deformed_samples):
                snip_deformed = deform(snip, start, deform_len, tau)
                snip = snip_deformed
                # fake data
                n_states = len(snip)
                home_state = snip[0]
                for idx in range(start, deform_len):
                    position = snip[idx]
                    traj_type = 1
                    dataset.append((home_state.tolist() + position.tolist(), position.tolist(), traj_type))
                    false_cnt += 1
                    # print(dataset[-1])
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[-1])
    print("[*] I have this many subtrajectories: ", len(dataset))
    print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))

    model = Net().to(device)
    dataname = 'data/' + 'class_' + str(tasks) + "_" + str(model_no) +'.pkl'
    savename = 'models/' + 'class_' + str(tasks) + "_" + str(model_no)

    EPOCH = 500
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.005
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    raw_data = random.sample(raw_data, len(raw_data))
    # raw_data = raw_data.tolist()
    inputs = [element[:2] for element in raw_data]
    targets = [element[2] for element in raw_data]
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
    num_tasks = 1
    train_classifier(num_tasks)

if __name__ == "__main__":
    main()
