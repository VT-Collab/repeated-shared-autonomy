import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

# collect dataset
class MotionData(Dataset):

    def __init__(self, filename):
        self.data = pickle.load(open(filename, "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        state = torch.FloatTensor(item[1]).to(device)
        true_z = torch.FloatTensor(item[2]).to(device)
        action = torch.FloatTensor(item[3]).to(device)
        return (snippet, state, true_z, action)


# conditional autoencoder
class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.loss_func = nn.MSELoss()

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(6, 10),
            nn.Tanh(),
            # nn.Linear(15, 10),
            # nn.Tanh(),
            nn.Linear(10, 1)
        )

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(4, 20),
            nn.Tanh(),
            nn.Linear(20, 15),
            nn.Tanh(),
            nn.Linear(15, 3)
        )

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        history, state, ztrue, action = x
        z = self.encoder(history)
        ztrue = torch.FloatTensor(ztrue).to(device)
        z_with_state = torch.cat((z, state), 1)
        action_decoded = self.decoder(z_with_state)
        loss = self.loss(action, action_decoded)
        return loss

    def loss(self, action_decoded, action_target):
        return self.loss_func(action_decoded, action_target)

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
    lookahead = 5
    noiselevel = 0.05
    noisesamples = 5

    notepad_count = 0
    tape_count = 0
    glass_count = 0
    shelf_count = 0
    cup_count = 0
    savename = 'data/' + '0_cae_' + str(tasks) + '.pkl'
    for filename in os.listdir(folder):
        demo = pickle.load(open(folder + "/" + filename, "rb"))
        traj = [item[0] for item in demo]
        # print(traj)
        n_states = len(traj)
        z = [1.0]
        for idx in range(n_states-lookahead):
            home_state = traj[idx][:3]
            position = traj[idx][3:]
            nextposition = traj[idx + lookahead][3:]
            for jdx in range(noisesamples):
                action = 20 * (nextposition - (position + np.random.normal(0, noiselevel, 3)))
                dataset.append((home_state + position, position, z, action.tolist()))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))

    model = CAE().to(device)
    dataname = 'data/' + '0_cae_' + str(tasks) + '.pkl'
    savename = 'models/' + '0_cae_' + str(tasks)

    EPOCH = 500
    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
    BATCH_SIZE_TRAIN = int(train_data.__len__() / 10.)
    # print(BATCH_SIZE_TRAIN)
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
