import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import sys
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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
            nn.Linear(14, 7),
            nn.Tanh(),
            # nn.Linear(10, 10),
            # nn.Tanh(),
            nn.Linear(7, 1)
        )

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(8, 30),
            nn.Tanh(),
            nn.Linear(30, 15),
            nn.Tanh(),
            nn.Linear(15, 7)
        )

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        history, state, ztrue, action = x
        z = self.encoder(history)
        z_with_state = torch.cat((z, state), 1)
        action_decoded = self.decoder(z_with_state)
        loss = self.loss(action, action_decoded)
        return loss

    def loss(self, action_decoded, action_target):
        return self.loss_func(action_decoded, action_target)

# train cAE
def train_cae(tasks):
    # tasks = int(sys.argv[1])
    # tasks = int(tasks)

    dataset = []
    folder = 'demos/robot'
    lookahead = 5
    noiselevel = 0.005
    noisesamples = 5

    savename = 'data/robot/' + 'cae_' + str(tasks) + '.pkl'
    for filename in os.listdir(folder):
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        # print(traj)
        n_states = len(traj)
        z = [1.0] # This is used to test the decoder capabilities
        # home_state = traj[0]
        for idx in range(n_states-lookahead):
            home_state = traj[idx][:7]
            position = np.asarray(traj[idx])[7:]
            nextposition = np.asarray(traj[idx + lookahead])[7:]
            for jdx in range(noisesamples):
                action = nextposition - (position + np.random.normal(0, noiselevel, 7))
                dataset.append((home_state + position.tolist(), position.tolist(), z, action.tolist()))

    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))

    model = CAE().to(device)
    dataname = 'data/robot/' + 'cae_' + str(tasks) + '.pkl'
    savename = 'models/robot/' + 'cae_' + str(tasks)

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

def main():
    num_tasks = "pushing"
    train_cae(num_tasks)


if __name__ == "__main__":
    main()
