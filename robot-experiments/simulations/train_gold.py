import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np

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

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(8, 30),
            nn.Tanh(),
            nn.Linear(30, 15),
            nn.Tanh(),
            nn.Linear(15, 7)
        )


    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        history, state, ztrue, action = x
        ztrue = torch.FloatTensor(ztrue).to(device)
        z_with_state = torch.cat((ztrue, state), 1)
        action_decoded = self.decoder(z_with_state)
        loss = self.loss(action, action_decoded)
        return loss

    def loss(self, action_decoded, action_target):
        return self.loss_func(action_decoded, action_target)


# train cAE
def main(model_num):

    model = CAE().to(device)
    dataname = 'data/task_1_cae_' + str(model_num) + '.pkl'
    savename = 'models/task_1_gold_' + str(model_num)

    EPOCH = 500
    if model_num == 1:
        BATCH_SIZE_TRAIN = 106
    elif model_num == 3:
        BATCH_SIZE_TRAIN = 342
    elif model_num == 5:
        BATCH_SIZE_TRAIN = 575

    LR = 0.01
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
    # BATCH_SIZE_TRAIN = int(train_data.__len__() / 10.)
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
    for i in range(1, 6, 2):
        main(i)
