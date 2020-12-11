import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


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
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(10, 12),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(10, 1)
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(9, 12),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        c = x[0]
        s = x[1]
        ztrue = x[2] #this is the ground truth label, for use when trouble shooting
        a = x[3]
        z = self.encoder(c)
        z_with_state = torch.cat((z, s), 1)
        a_decoded = self.decoder(z_with_state)
        loss = self.loss(a, a_decoded)
        return loss

    def loss(self, a_decoded, a_target):
        return self.loss_func(a_decoded, a_target)


# train cAE
def main():

    model = CAE()
    model = model.to(device)
    dataname = 'data/dataset.pkl'
    savename = "models/cae_gpu"

    EPOCH = 2000
    BATCH_SIZE_TRAIN = 400
    LR = 0.01
    LR_STEP_SIZE = 1400
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
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
