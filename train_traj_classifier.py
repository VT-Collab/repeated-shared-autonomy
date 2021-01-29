import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

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


# Classifier
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(10, 12),
            nn.Tanh(),
            # nn.Linear(14, 14),
            # nn.Tanh(),
            nn.Linear(12, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 2),
        )

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        c = x[0]
        s = x[1]
        y_output = self.classify(c)
        y_true = x[4]
        loss = self.loss(y_output, y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)


# train classifier
def main():

    model = Net()
    model = model.to(device)
    dataname = 'data/dataset_med_sigma.pkl'
    savename = "models/classifier_traj"

    EPOCH = 1500
    BATCH_SIZE_TRAIN = 400
    LR = 0.006 # Learning rate
    LR_STEP_SIZE = 1400 # Num epochs before LR decay
    LR_GAMMA = 0.1 # LR decay factor

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
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)

if __name__ == "__main__":
    main()