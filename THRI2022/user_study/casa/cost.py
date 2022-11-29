import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import rospy, argparse
from glob import glob
from utils import TrajectoryClient, convert_to_6d
from geometry_msgs.msg import Twist

np.set_printoptions(precision=2, suppress=True)

device = "cpu"

# collect dataset
class MotionData(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        state = torch.FloatTensor(item[1]).to(device)
        action = torch.FloatTensor(item[2]).to(device)
        return (snippet, state, action)


# conditional autoencoder
class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.loss_func = nn.MSELoss()

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(18, 30),
            nn.Tanh(),
            nn.Linear(30, 40),
            nn.Tanh(),
            nn.Linear(40, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 6)
        )

    def encoder(self, x):
        return self.enc(x)

    def forward(self, x):
        history, state, action = x
        z =  self.encoder(history)
        loss = self.loss(action, z)
        return loss

    def loss(self, action_decoded, action_target):
        # return self.loss_func(action_decoded, action_target)
        return self.loss_func(action_decoded[:,:3], action_target[:,:3]) + .9 * self.loss_func(action_decoded[:,3:], action_target[:,3:])

def train_cae(args):
    mover = TrajectoryClient()

    parent_folder = '../demos'


    lookahead = args.lookahead#5
    noiselevel = args.noiselevel#0.0005
    noisesamples = args.noisesamples#5
    intentfolder = args.intentfolder 
    folders = [intentfolder]
    rospy.loginfo("Using demos for tasks : {}".format(folders))
    demos = []
    dataset = []
    data_folder = "data"
    model_folder = "models"
    savename = 'cae_' + "_".join(folders)

    for folder in folders:
        demos += glob(parent_folder + "/" + folder + "/*.pkl")

    inverse_fails = 0
    for filename in demos:
        demo = pickle.load(open(filename, "rb"))
        n_states = len(demo)

        for idx in range(n_states-lookahead):

            home_pos = np.asarray(demo[idx]["start_pos"])
            home_q = np.asarray(demo[idx]["start_q"])
            home_gripper_pos = [demo[idx]["start_gripper_pos"]]
            
            curr_pos = np.asarray(demo[idx]["curr_pos"])
            curr_q = np.asarray(demo[idx]["curr_q"])
            curr_gripper_pos = [demo[idx]["curr_gripper_pos"]]
            curr_trans_mode = [float(demo[idx]["trans_mode"])]
            curr_slow_mode = [float(demo[idx]["slow_mode"])]

            next_pos = np.asarray(demo[idx+lookahead]["curr_pos"])
            next_q = np.asarray(demo[idx+lookahead]["curr_q"])
            next_gripper_pos = [demo[idx+lookahead]["curr_gripper_pos"]]

            for _ in range(noisesamples):
                # add noise in cart space
                noise_pos = curr_pos.copy() + np.random.normal(0, noiselevel, len(curr_pos))
                
                # convert to twist for kdl_kin
                noise_pos_twist = Twist()
                noise_pos_twist.linear.x = noise_pos[0]
                noise_pos_twist.linear.y = noise_pos[1]
                noise_pos_twist.linear.z = noise_pos[2]
                noise_pos_twist.angular.x = noise_pos[3]
                noise_pos_twist.angular.y = noise_pos[4]
                noise_pos_twist.angular.z = noise_pos[5]

                noise_q = np.array(mover.pose2joint(noise_pos_twist, guess=curr_q))
                # noise_q = curr_q

                if None in noise_q:
                    inverse_fails += 1
                    continue

                # Angle wrapping is a bitch
                noise_pos_awrap = convert_to_6d(noise_pos)
                # next_pos_awrap = convert_to_6d(next_pos)

                action = next_pos - noise_pos

                # history = noise_q + noise_pos + curr_gripper_pos + trans_mode + slow_mode
                history = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                state = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                dataset.append((history, state, action.tolist()))

    if inverse_fails > 0:
        rospy.loginfo("Failed inverses: {}".format(inverse_fails))

    pickle.dump(dataset, open(data_folder + "/" + savename, "wb"))

    model = CAE().to(device)

    EPOCH = 50
    BATCH_SIZE_TRAIN = 2#int(train_data.__len__() / 10.)
    LR = 0.0001
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.15

    train_data = MotionData(dataset)
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
        rospy.loginfo("epoch: {} loss: {}".format(epoch, loss.item()))
        torch.save(model.state_dict(), model_folder + "/" + savename)

def main():
    rospy.init_node("train_cae")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, help="number of tasks to use", default=1)
    parser.add_argument("--lookahead", type=int, help="lookahead to compute robot action", default=5)
    parser.add_argument("--noisesamples", type=int, help="num of noise samples", default=5)
    parser.add_argument("--noiselevel", type=float, help="variance for noise", default=.0005)
    parser.add_argument("--intentfolder", help="folder for intent", default="pour")
    args = parser.parse_args()
    train_cae(args)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass