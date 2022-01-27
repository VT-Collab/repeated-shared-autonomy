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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        true_z = torch.FloatTensor(item[2]).to(device)
        label = self.target[idx]
        return (snippet, state, true_z, label)


# conditional autoencoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

        # Encoder
        self.classifier = nn.Sequential(
            nn.Linear(9, 15),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(15, 20),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(20, 15),
            nn.Tanh(),
            nn.Linear(15, 7),
            nn.Tanh(),
            nn.Linear(7, 2)
        )

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        c = x[0]
        s = x[1]
        y_output = self.classify(c)
        y_true = x[3]
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
    R = np.linalg.inv(A.T @ A)
    U = np.zeros(length)
    gamma = np.zeros((length, 3))
    for idx in range(3):
        U[0] = tau[idx]
        gamma[:,idx] = R @ U
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end,:] += gamma[0:end-start,:]
    return xi1

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

# train cAE
def main():

    tasks = int(sys.argv[1])

    dataset = []
    folder = 'demos'
    lookahead = 0
    noiselevel = 0.00
    # noisesamples = 3

    true_cnt = 0
    false_cnt = 0

    notepad_count = 0
    soupcan_count = 0
    tape_count = 0
    cup_count = 0
    traj_cnt = 0
    shelf_count = 0

    savename = 'data/' + '0_class_' + str(tasks) + '.pkl'
    deformed_trajs = []
    for filename in os.listdir(folder):
        if filename[0] == ".":
            continue
        demo = pickle.load(open(folder + "/" + filename, "rb"))
        traj = [item[0] for item in demo]
        action = [item[1] for item in demo]
        n_states = len(traj)
        z = [1.0]
        for idx in range(n_states):
            home_state = traj[idx][:3]
            position = traj[idx][3:]
            traj_type = 0
            dataset.append((home_state + position + action[idx], position, z, action[idx], traj_type))
            true_cnt += 1
        snippets = np.array_split(traj, 1)
        deformed_samples = 2
        for snip in snippets:
            tau = np.random.uniform([-0.1]*3, [0.1]*3)
            deform_len = len(snip)
            # print(deform_len)
            start = 0
            for i in range(deformed_samples):
                snip_deformed = deform(snip[:,3:], 0, deform_len, tau)
                deformed_trajs.append(snip_deformed)
                snip[:,3:] = snip_deformed
                # fake data
                n_states = len(snip)
                for idx in range(start, deform_len):
                    home_state = snip[idx][:3].tolist()
                    position = snip[idx][3:] 
                    #position = position + np.random.normal(0, noiselevel, 7)
                    traj_type = 1
                    dataset.append((home_state + position.tolist() + action[idx], position.tolist(), z, action[idx], traj_type))
                    false_cnt += 1
                    # print(dataset[-1])
    pickle.dump(deformed_trajs, open("deformed_trajs.pkl", "wb"))
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[-1])
    print("[*] I have this many subtrajectories: ", len(dataset))
    print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))

    model = Net().to(device)
    dataname = 'data/' + '0_class_' + str(tasks) + '.pkl'
    savename = 'models/' + '0_class_' + str(tasks)

    EPOCH = 500
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.005
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    raw_data = random.sample(raw_data, len(raw_data))
    # raw_data = raw_data.tolist()
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


if __name__ == "__main__":
    main()