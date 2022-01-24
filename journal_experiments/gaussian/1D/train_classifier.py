import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from sklearn.utils import shuffle
import sys
import os
import copy 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

# collect dataset
class MotionData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.target = y
        # self.data, self.target = shuffle(self.data, self.target)
        self.target = torch.as_tensor(self.target).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item).to(device)
        label = self.target[idx]
        return (snippet, label)


# conditional autoencoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

        self.classifier = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 2)
        )

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        c = x[0]
        y_output = self.classify(c)
        y_true = x[-1]
        loss = self.loss(y_output, y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)

def main():
    tasks = int(sys.argv[1])

    dataset = []
    folder = 'demos'

    true_cnt = 0
    false_cnt = 0
    savename = 'data/' + '0_class_' + str(tasks) + '.pkl'
    for filename in os.listdir(folder):
        if filename[0] != ".":
            demo = pickle.load(open(folder + "/" + filename, "rb"))
            traj_type = 0
            print(demo)
            for pair in demo:
                dataset.append((pair, traj_type))
                true_cnt += 1
            deformed_samples = 10
            for i in range(deformed_samples):
                for pair in demo:
                    s = np.copy(pair)
                    s[1] += np.random.normal(0.25, 5)
                    traj_type = 1
                    dataset.append((s.tolist(), traj_type))
                    false_cnt += 1
            # print(dataset[-1])
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[-1])
    print("[*] I have this many subtrajectories: ", len(dataset))
    print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))

    model = Net().to(device)
    dataname = 'data/' + '0_class_' + str(tasks) + '.pkl'
    savename = 'models/' + '0_class_' + str(tasks)

    EPOCH = 500
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.005
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    raw_data = random.sample(raw_data, len(raw_data))
    # raw_data = raw_data.tolist()
    inputs = [element[0] for element in raw_data]
    targets = [element[1] for element in raw_data]
    # print(inputs[0])

    train_data = MotionData(inputs, targets)
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
