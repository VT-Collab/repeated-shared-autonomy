import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# collect dataset
class MotionData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.target = torch.as_tensor(y).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        state = torch.FloatTensor(item[1]).to(device)
        true_z = torch.FloatTensor(item[2]).to(device)
        action = torch.FloatTensor(item[3]).to(device)
        label = self.target[idx]
        return (snippet, state, true_z, action, label)


# conditional autoencoder
class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.loss_func = nn.MSELoss()
        self.BETA = 0.001
        self.lamda = 0.001

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(10, 10),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(10, 12),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.Tanh(),
            # nn.Dropout(0.1)
            nn.Linear(10, 1)
        )
        # self.fc_mean = nn.Linear(10, 1)
        # self.fc_var = nn.Linear(10, 1)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(9, 12),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(10, 4)
        )
        self.dec_mean = nn.Linear(4, 2)
        self.dec_var = nn.Linear(4, 2)

    def reparam(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        a = self.dec(z_with_state)
        return self.dec_mean(a), self.dec_var(a)

    def forward(self, x):
        c = x[0]
        s = x[1]
        ztrue = x[2] #this is the ground truth label, for use when trouble shooting
        a = x[3]
        y_true = x[4]
        z = self.encoder(c)
        z_with_state = torch.cat((z, s), 1)
        a_mean, a_var = self.decoder(z_with_state)
        a_decoded = self.reparam(a_mean, a_var)
        loss = self.loss(a, a_decoded, a_mean, a_var, y_true)
        return loss

    def loss(self, a_decoded, a_target, mean, log_var, y_true):
       
        # Change true taj from 0 to 1
        y_true_flipped = torch.abs(y_true - 1)

        # Set fake traj actions to zero
        a_decoded = y_true_flipped.unsqueeze(1) * a_decoded 
        a_target = y_true_flipped.unsqueeze(1) * a_target
        mean = y_true_flipped.unsqueeze(1) * mean
        log_var_flipped = y_true_flipped.unsqueeze(1) * log_var

        rce = self.loss_func(a_decoded, a_target)
        # kld = -0.5 * torch.sum(1 + log_var_flipped - mean.pow(2) - log_var_flipped.exp())
        action_loss = rce# + kld
        entropy = torch.sum(log_var)

        # Loss = MSE(action_true - policy_true) - entropy(fake)
        total_loss = action_loss - self.lamda * entropy

        return total_loss, action_loss, entropy 

# train cAE
def main(num):

    model = CAE()
    model = model.to(device)
    dataname = 'data/dataset_with_fake_dist.pkl'
    savename = "models/entropy_0001_" + str(num)
    print(savename)

    EPOCH = 1750
    BATCH_SIZE_TRAIN = 400
    LR = 0.01
    LR_STEP_SIZE = 1400
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    inputs = [element[:4] for element in raw_data]
    targets = [element[4] for element in raw_data]

    train_data = MotionData(inputs, targets)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss, action_loss, entropy = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("Epoch: {0:4d}    Loss: {1:2.3f}  Action_Loss: {2:2.3f}   Entropy: {3:2.3f}".\
                    format(epoch, loss.item(), action_loss.item(), entropy.item()))
    torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    for i in range(1,2):
        main(i)
