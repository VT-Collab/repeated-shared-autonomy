import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from torch.autograd import Variable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Some constants
ALPHA = 1.
BETA = 1.
TAU = 1.0
T = 0.033 #time between data points (30fps => 1/30 sec)

# collect dataset
class MotionData(Dataset):

    def __init__(self, filename):
        self.all_data = pickle.load(open(filename, "rb"))
        self.max_len = len(self.all_data)
        self.train_len = int(0.75 * self.max_len)
        self.data = random.sample(self.all_data, self.train_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.all_data[idx]
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
        # self.ALPHA = 0.1
        # self.BETA = 0.01
        # self.TAU = 0.1
        # self.t = 0.033 #time between data points (30fps => 1/30 sec)
        self.prev_y = torch.zeros(400, 2)
        self.prev_y_dot = torch.zeros(400, 2)

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(10, 12),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.ReLU(),
            # nn.Dropout(0.1)
            nn.Linear(10, 2)
        )

    def encoder(self, x):
        return self.enc(x)

    def forward(self, x):
        c = x[0]
        s = x[1]
        ztrue = x[2] #this is the ground truth label, for use when trouble shooting
        a = x[3]
        z = self.encoder(c)
        a_decoded = self.action(z)
        loss = self.loss(a_decoded, a)
        return loss

    def loss(self, a_decoded, a_target):
        return self.loss_func(a_decoded, a_target)

    def action(self, z):
        y_ddot = torch.zeros(z.size())
        y_dot = torch.zeros(z.size())
        y = torch.zeros(z.size())

        y_ddot = 1 / TAU * ALPHA * (BETA * (z - self.prev_y) - self.prev_y_dot)
        y_dot = self.prev_y_dot + y_ddot * T
        y = self.prev_y + y_dot * T + 0.5 * y_ddot * T ** 2
        
        self.prev_y = y
        self.prev_y_dot = y_dot
        return y

    def reset(self):
        self.prev_y = torch.zeros(400, 2)
        self.prev_y_dot = torch.zeros(400, 2)

# train cAE
def main(num):

    model = CAE()
    model = model.to(device)
    dataname = 'data/dataset.pkl'
    savename = "models/dmp_" + str(num)
    print(savename)

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
            a_target = x[3]
            optimizer.zero_grad()
            loss = model(x)
            loss.detach_()
            loss = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
    torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    for i in range(1,2):
        main(i)
