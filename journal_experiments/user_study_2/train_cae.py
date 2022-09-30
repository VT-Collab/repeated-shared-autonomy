import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import sys
import os, tf
from glob import glob
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(16, 30),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Tanh(),
            # nn.Linear(40, 30),
            # nn.Tanh(),
            nn.Linear(20, 20)
        )

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(29, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Tanh(),
            # nn.Linear(20, 20),
            # nn.Tanh(),
            nn.Linear(20, 6)
        )

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        history, state, ztrue, action = x
        z =  self.encoder(history)
        z_with_state = torch.cat((z, state), 1)
        action_decoded = self.decoder(z_with_state)

        loss = self.loss(action[:, :3], action_decoded[:, :3]) + 2 * self.loss(action[:, 3:], action_decoded[:, 3:])
        return loss

    def loss(self, action_decoded, action_target):
        return self.loss_func(action_decoded, action_target)

# train cAE
def train_cae():
    dataset = []
    parent_folder = 'demos/old_home_pos'
    lookahead = 5
    noiselevel = 0.005
    noisesamples = 5

    savename = 'data/' + 'cae.pkl'
    # folders = ["pour", "stir", "place"]
    folders = ["pour", "place"]
    demos = []
    for folder in folders:
        demos += glob(parent_folder + "/" + folder + "/*.pkl")
    print(demos)
    for filename in demos:
        demo = pickle.load(open(filename, "rb"))
        traj = [item[0] + item[1] + item[2] + item[3] + item[4]+ item[5] for item in demo]
        # print(traj)
        n_states = len(traj)
        # if filename[0] == '1':
        #     z = [1.0] # This is used to test the decoder capabilities
        # elif filename[0] == '2':
        #     z = [-1.0]
        z = [1.0]
        # home_state = traj[0]
        for idx in range(1, n_states-lookahead):
            home_state = np.asarray(traj[idx][:7])
            # home_state = np.zeros(8)
            # home_state[:3] = traj[idx][:3]
            # home_state[3:7]= tf.transformations.quaternion_from_euler(traj[idx][3], traj[idx][4], traj[idx][5])
            # home_state[7] = traj[idx][6]
            prev_pos = np.asarray(traj[idx-1])[7:14]
            position = np.asarray(traj[idx])[7:14]
            # print(np.linalg.norm(position - np.asarray(traj[idx+2][7:14])))
            # position = np.zeros(8)
            # position[:3] = traj[idx][7:10]
            # position[3:7]= tf.transformations.quaternion_from_euler(traj[idx][10], traj[idx][11], traj[idx][12])
            # position[7] = traj[idx][13]
            # nextposition = np.zeros(8)
            # nextposition[:3] = traj[idx + lookahead][7:10]
            # nextposition[3:7]= tf.transformations.quaternion_from_euler(traj[idx + lookahead][10], traj[idx + lookahead][11], traj[idx + lookahead][12])
            # nextposition[7] = traj[idx + lookahead][13]
            nextposition = np.asarray(traj[idx + lookahead])[7:14]
            for jdx in range(noisesamples):
                noise_position = position.copy()
                noise_position[:6] = position[:6] + np.random.normal(0, noiselevel, len(position[:6]))
                action = nextposition - noise_position #(position + np.random.normal(0, noiselevel, 6))
                # dataset.append((home_state.tolist() + noise_position.tolist() + traj[idx][14:], noise_position.tolist()+ traj[idx][14:], z, action.tolist()))
                dataset.append((prev_pos.tolist() + noise_position.tolist() + traj[idx][14:], noise_position.tolist()+ traj[idx][14:], z, action[:6].tolist()))
        # sys.exit()
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))

    model = CAE().to(device)
    dataname = 'data/' + 'cae.pkl'
    savename = 'models/' + 'cae'

    EPOCH = 500
    LR = 0.0001
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.15

    train_data = MotionData(dataname)
    BATCH_SIZE_TRAIN = int(train_data.__len__() / 10.)
    # print(BATCH_SIZE_TRAIN)
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

def main():
    train_cae()


if __name__ == "__main__":
    main()
