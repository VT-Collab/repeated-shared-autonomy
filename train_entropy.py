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
        self.lamda = 10

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
            # nn.Linear(10, 10)
        )
        self.fc_mean = nn.Linear(10, 1)
        self.fc_var = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh()
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(9, 12),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(10,10),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(10, 2)
        )
        self.dec_mean = nn.Linear(4, 2)
        self.dec_var = nn.Sequential(
            nn.Linear(4, 2),
            nn.Tanh()
        )

    def reparam(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encoder(self, x):
        # return self.enc(x)
        h = self.enc(x)
        return self.fc_mean(h), self.fc_var(h)

    def decoder(self, z_with_state):
        # a = self.dec(z_with_state)
        # return self.dec_mean(a), self.dec_var(a)
        return self.dec(z_with_state)

    def forward(self, x):
        c = x[0]
        s = x[1]
        ztrue = x[2] #this is the ground truth label, for use when trouble shooting
        a = x[3]
        y_true = x[4]
        # z = self.encoder(c)
        # z_with_state = torch.cat((z, s), 1)
        # mean, log_var = self.decoder(z_with_state)
        # a_decoded = self.reparam(mean, log_var)
        mean, log_var = self.encoder(c)
        z = self.reparam(mean, log_var)
        z_with_state = torch.cat((z, s), 1)
        a_decoded = self.decoder(z_with_state)
        loss = self.loss(a, a_decoded, mean, log_var, y_true)
        return loss

    def loss(self, a_decoded, a_target, mean, log_var, y_true):
        # Change true taj from 0 to 1
        true_traj_idxs = torch.nonzero(y_true - 1)
        false_traj_idxs = torch.nonzero(y_true)

        a_decoded_true = a_decoded[true_traj_idxs]
        a_target_true = a_target[true_traj_idxs]
        mean_true = mean[true_traj_idxs]
        log_var_true = log_var[true_traj_idxs]

        entropy_true = torch.sum(torch.abs(log_var_true))
        
        log_var_false = log_var[false_traj_idxs]

        rce = self.loss_func(a_decoded_true, a_target_true)
        kld = -0.5 * torch.sum(1 + log_var_true - mean_true.pow(2) - log_var_true.exp())
        action_loss = rce + self.BETA * kld
        entropy_false = torch.sum(torch.abs(log_var_false))

        # Loss = MSE(action_true - policy_true) - entropy(fake)
        total_loss = action_loss + .0001 * entropy_true + self.lamda / entropy_false

        return total_loss, action_loss, entropy_true, entropy_false 

# train cAE
def main(num):

    model = CAE()
    model = model.to(device)
    dataname = 'data/dataset_with_fake_dist.pkl'
    savename = "models/entropy_" + str(num)
    print(savename)

    EPOCH = 2500
    BATCH_SIZE_TRAIN = 400
    LR = 0.001
    LR_STEP_SIZE = 2250
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
            loss, action_loss, entropy_true, entropy_false = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print(\
        "Epoch: {0:4d}    Loss: {1:2.3f}  A_Loss: {2:2.3f}   E_True: {3:2.1f}   E_False: {4:2.1f}".\
            format(epoch, loss.item(), action_loss.item(), entropy_true.item(), entropy_false.item()))
    torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    for i in range(1,2):
        main(i)
