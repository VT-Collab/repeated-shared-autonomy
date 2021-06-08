import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU memory from previous runs
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
        self.BETA = 0.1

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(18, 15),
            nn.Tanh(),
            nn.Linear(15, 10),
            nn.Tanh(),
            # nn.Linear(10, 1)
        )
        self.fc_mean = nn.Linear(10,1)
        self.fc_var = nn.Linear(10,1)

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(12, 30),
            nn.Tanh(),
            nn.Linear(30, 15),
            nn.Tanh(),
            nn.Linear(15, 7)
        )

    def reparam(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encoder(self, x):
        h = self.enc(x)
        return self.fc_mean(h), self.fc_var(h)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        history, state, ztrue, action = x
        mean, log_var = self.encoder(history)
        z = self.reparam(mean, log_var)
        z_with_state = torch.cat((z, state), 1)
        action_decoded = self.decoder(z_with_state)
        loss = self.loss(action, action_decoded, mean, log_var)
        return loss

    def loss(self, action_decoded, action_target, mean, log_var):
        rce = self.loss_func(action_decoded, action_target)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return rce + self.BETA * kld, rce


# train cAE
def main():

    model = CAE().to(device)
    dataname = 'data/dataset.pkl'
    savename = "models/vae_model_r"

    EPOCH = 500
    BATCH_SIZE_TRAIN = 10000
    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss, error = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item(), error.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
