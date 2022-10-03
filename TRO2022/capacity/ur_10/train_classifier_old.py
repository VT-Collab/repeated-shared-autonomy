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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

# collect dataset
class MotionData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.target = y
        # self.data, self.target = shuffle(self.data, self.target)
        # self.target = torch.as_tensor(self.target).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item)
        label = torch.FloatTensor(self.target[idx])
        return (snippet, label)
        # snippet = torch.FloatTensor(item[0]).to(device)
        # state = torch.FloatTensor(item[1]).to(device)
        # label = self.target[idx]
        # return (snippet, state, label)


# conditional autoencoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([20., 1.]))

        # Encoder
        self.classifier = nn.Sequential(
            nn.Linear(18, 30),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(30, 15),
            nn.Tanh(),
            # nn.Dropout(0.1),
            # nn.Linear(30, 15),
            # nn.Tanh(),
            nn.Linear(15, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
            # nn.Linear(14, 7),
            # nn.Tanh(),
            # # nn.Dropout(0.1),
            # # nn.Linear(14, 20),
            # # nn.Tanh(),
            # # nn.Dropout(0.1),
            # # nn.Linear(10, 10),
            # # nn.Tanh(),
            # nn.Linear(7, 2)
        )

    def classify(self, x):
        return self.classifier(x)

    # def forward(self, x):
    #     c = x[0]
    #     s = x[1]
    #     y_output = self.classify(c)
    #     y_true = x[2]
    #     loss = self.loss(y_output, y_true)
    #     return loss
    def forward(self, x):
        c = x[0]
        y_output = self.classify(c)
        y_true = x[1]
        loss = self.loss(y_output, y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)


def deform(xi, start, length, tau):
    length += np.random.choice(np.arange(50, length))
    xi1 = copy.deepcopy(np.asarray(xi))
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(np.dot(A.T, A))
    U = np.zeros(length)
    gamma = np.zeros((length, 6))
    for idx in range(6):
        U[0] = tau[idx]
        gamma[:,idx] = np.dot(R, U)
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end+1,:] += gamma[0:end-start+1,:]
    return xi1

# train cAE
def train_classifier(tasklist, max_demos, model_num):
    tasks = []
    for task in tasklist:
        for i in range(1, max_demos+1):
            tasks.append(task + "_" + str(i) + ".pkl")
    # tasks = sys.argv[1]
    # tasks = int(tasks)

    dataset = []
    folder = 'demos'
    lookahead = 0
    noiselevel = 0.05
    deformed_trajs = []
    # noisesamples = 3
    z = [1.0]

    true_cnt = 0
    false_cnt = 0
    savename = 'data/' + 'class_' + "_".join(tasklist) + "_" + str(model_num) + '_old.pkl'
    for filename in os.listdir(folder):
        if not filename in tasks:
            continue
        demo = pickle.load(open(folder + "/" + filename, "rb"))

        # Human's demonstrations
        data = [item[0] + item[1] for item in demo]
        # traj = torch.Tensor(traj).unsqueeze(0)
        label = np.zeros(len(demo)).tolist()
        # label = 0.
        dataset.append([data, label])

        # old method
        # traj = [item[0] for item in demo]
        # action = [item[1] for item in demo]
        # n_states = len(traj)

        # for idx in range(n_states):
        #     home_state = traj[idx][:6]
        #     position = traj[idx][6:]
        #     traj_type = 0
        #     dataset.append((home_state + position + action[idx], position, z, action[idx], traj_type))
        #     true_cnt += 1

        # snippets = np.array_split(traj, 1)
        # deformed_samples = 10
        # for snip in snippets:
        #         tau = np.random.uniform([-0.00035]*6, [-0.00035]*6)
        #         deform_len = len(snip)
        #         # print(deform_len)
        #         for i in range(deformed_samples):
        #             # print(snip[:,6:9])
        #             start = np.random.choice(np.arange(20, int(deform_len/2)))
        #             snip_deformed = deform(snip[:,6:], start, deform_len, tau)
        #             snip[:,6:] = snip_deformed
        #             # fake data
        #             n_states = len(snip)
        #             for idx in range(start, deform_len):
        #                 home_state = snip[idx][:6].tolist()
        #                 position = snip[idx][6:] 
        #                 traj_type = 1
        #                 dataset.append((home_state + position.tolist() + action[idx], position.tolist(), z, action[idx], traj_type))
        #                 false_cnt += 1
    pickle.dump(dataset, open(savename, "wb"))
    # print(dataset[-1])
    # print("[*] I have this many subtrajectories: ", len(dataset))
    # print("[*] false count: " + str(false_cnt) + " true: " + str(true_cnt))

    model = Net().to(device)
    dataname = 'data/' + 'class_' + "_".join(tasklist) + "_" + str(model_num) + '_old.pkl'
    savename = 'models/' + 'class_' + "_".join(tasklist) + "_" + str(model_num) + "_old"

    EPOCH = 500
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.01
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    raw_data = random.sample(raw_data, len(raw_data))
    # raw_data = raw_data.tolist()
    inputs = [element[0] for element in raw_data]
    targets = [element[1] for element in raw_data]
    # print(inputs[0])

    train_data = MotionData(inputs, targets)
    BATCH_SIZE_TRAIN = int(train_data.__len__() / 5.)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            network_inputs = []
            network_labels = []
            for dataset in x:
                traj = x[0][0].numpy()
                labels = x[1][0].numpy()

                network_inputs += traj.tolist()
                network_labels += labels.tolist()

                deformed_samples = 20
                tau = np.random.uniform([-0.00045]*6, [-0.00045]*6)
                deform_len = len(traj)
                for i in range(deformed_samples):
                    start = np.random.choice(np.arange(20, 100))
                    snip_deformed = deform(traj[:,6:12], start, deform_len, tau)
                    traj[:,6:12] = snip_deformed
                    labels = np.ones(deform_len)
                    network_inputs += traj.tolist()
                    network_labels += labels.tolist()
            network_inputs = torch.FloatTensor(network_inputs)
            network_labels = torch.LongTensor(network_labels)
            r_idx = torch.randperm(len(network_inputs))
            network_inputs = network_inputs[r_idx, :]
            network_labels = network_labels[r_idx]
            x = [network_inputs, network_labels]
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)

def main():
    required_tasks = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
                      ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
                      ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
                      ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]
    # required_tasks = [["push2"], ["cut1"], ["cut2"], ["scoop1"], ["scoop2"], ["open1"], ["open2"]]
    max_demos = 15
    max_models = 20
    for model_num in range(3, max_models):
        for tasklist in required_tasks:
            print("[*] Training for task: ", tasklist)
            train_classifier(tasklist, max_demos, model_num)

if __name__ == "__main__":
    main()
