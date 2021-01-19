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


# conditional autoencoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(10, 10),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(10, 12),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(12, 10),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(10, 2)
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


# train cAE
def main():

    model = Net()
    model = model.to(device)
    dataname = 'data/dataset_with_fake_dist.pkl'
    savename = "models/classifier_dist"

    EPOCH = 2000
    BATCH_SIZE_TRAIN = 400
    LR = 0.01
    LR_STEP_SIZE = 1400
    LR_GAMMA = 0.1

    raw_data = pickle.load(open(dataname, "rb"))
    inputs = [element[:4] for element in raw_data]
    targets = [element[4] for element in raw_data]

    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, stratify=targets)

    pickle.dump(X_train, open("data/X_train_data.pkl", "wb"))
    pickle.dump(y_train, open("data/y_train_data.pkl", "wb"))
    pickle.dump(X_test, open("data/X_test_data.pkl", "wb"))
    pickle.dump(y_test, open("data/y_test_data.pkl", "wb"))

    train_data = MotionData(X_train, y_train)
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

    test_outputs = []
    correct = 0
    with torch.no_grad():
        for idx, data in enumerate(X_test):
            c = torch.FloatTensor(data[0]).to(device)
            out = model.classify(c)
            _, label = torch.max(out.data, 0)
            test_outputs.append(label.item())
            if label.item() == y_test[idx]:
                correct += 1

    print(correct/float(len(y_test)))
if __name__ == "__main__":
    main()