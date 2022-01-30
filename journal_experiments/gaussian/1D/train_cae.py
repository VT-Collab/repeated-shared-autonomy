import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import sys
import os
from record_demos import GOAL, SIGMA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

# collect dataset
class MotionData(Dataset):

    def __init__(self, filename):
        self.dataset = pickle.load(open(filename, "rb"))
        self.data = self.dataset["dataset"]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        state = torch.as_tensor([item[0], 1]).to(device)
        action = torch.as_tensor(item[1]).to(device)
        return (state, action)


# conditional autoencoder
class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.loss_func = nn.MSELoss()

        # Encoder
        self.enc = nn.Sequential(
            # nn.Linear(2, 5),
            # nn.Tanh(),
            # nn.Linear(15, 10),
            # nn.Tanh(),
            nn.Linear(2, 1)
        )

        # Policy
        self.dec = nn.Sequential(
            # nn.Linear(3, 3),
            # nn.Tanh(),
            # nn.Linear(5, 5),
            # nn.Tanh(),
            nn.Linear(3, 1)
        )

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        state, action = x
        z = self.encoder(state)
        z_with_state = torch.cat((z, state), 1)
        action_decoded = self.decoder(z_with_state)
        loss = self.loss(action, action_decoded)
        return loss

    def loss(self, action_decoded, action_target):
        return self.loss_func(action_decoded, action_target)

# train cAE
def main():
    dataset = []
    folder = 'demos'

    savename = 'data/' + 'cae_' + str(GOAL) + '.pkl'
    for filename in os.listdir(folder):
        if filename[0] != ".":
            dataset = pickle.load(open(folder + "/" + filename, "rb"))
    data = {}
    data["goal"] = GOAL
    data["sigma"] = SIGMA
    data["dataset"] = dataset
    pickle.dump(data, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))

    model = CAE().to(device)
    dataname = 'data/' + 'cae_' + str(GOAL) + '.pkl'
    savename = 'models/' + 'cae_' + str(GOAL)

    EPOCH = 500
    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
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
