import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import sys
import rospy
from glob import glob
from utils import TrajectoryClient
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=2, suppress=True)

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
            nn.Linear(18, 30),
            nn.Tanh(),
            nn.Linear(30, 40),
            nn.Tanh(),
            nn.Linear(40, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 40)
        )

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(58, 40),
            nn.Tanh(),
            nn.Linear(40, 30),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
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

        loss = self.loss(action, action_decoded)
        return loss

    def loss(self, action_decoded, action_target):
        # return self.loss_func(action_decoded, action_target)
        return self.loss_func(action_decoded[:,:3], action_target[:,:3]) + .9 * self.loss_func(action_decoded[:,3:], action_target[:,3:])

def get_rotation_mat(euler):
    R_x = np.mat([[1, 0, 0],
                  [0, np.cos(euler[0]), -np.sin(euler[0])],
                  [0, np.sin(euler[0]), np.cos(euler[0])]])

    R_y = np.mat([[np.cos(euler[1]), 0, np.sin(euler[1])],
                  [0, 1, 0],
                  [-np.sin(euler[1]), 0, np.cos([1])]])

    R_z = np.mat([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                  [np.sin(euler[2]), np.cos(euler[2]), 0],
                  [0, 0, 1]])
    R = R_x * R_y * R_z
    return R
# train cAE
def train_cae():
    dataset = []
    parent_folder = 'demos'
    lookahead = 5
    noiselevel = 0.0005
    noisesamples = 5

    savename = 'data/' + 'cae.pkl'
    folders = ["pour", "stir", "place"]
    # folders = ["pour", "place"]
    # folders = ["stir"]
    demos = []

    mover = TrajectoryClient()
    for folder in folders:
        demos += glob(parent_folder + "/" + folder + "/*.pkl")
    print(demos)
    inverse_fails = 0
    for filename in demos:
        demo = pickle.load(open(filename, "rb"))

        n_states = len(demo)

        z = [1.0]

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

                noise_pos_twist = Twist()
                noise_pos_twist.linear.x = noise_pos[0]
                noise_pos_twist.linear.y = noise_pos[1]
                noise_pos_twist.linear.z = noise_pos[2]
                noise_pos_twist.angular.x = noise_pos[3]
                noise_pos_twist.angular.y = noise_pos[4]
                noise_pos_twist.angular.z = noise_pos[5]
                noise_q = np.array(mover.pose2joint(noise_pos_twist, guess=curr_q))
                # if 1 or not curr_trans_mode[0]:
                #     quat = quaternion_from_euler(noise_pos[3], noise_pos[4], noise_pos[5])
                #     before = noise_pos.copy()
                # for i in range(3):
                #     ang = noise_pos[3 + i]
                #     ang = math.fmod(ang, np.pi)
                #     if ang < 0:
                #         ang += np.pi
                #     if ang < 0.15:
                #         ang += np.pi
                #     noise_pos[3+i] = ang
                #     ang = next_pos[3 + i]
                #     ang = math.fmod(ang, np.pi)
                #     if ang < 0:
                #         ang += np.pi
                #     if ang < 0.15:
                #         ang += np.pi
                #     next_pos[3+i] = ang
                # 
                    # print("before: {}, after: {}".format(before[3:], noise_pos[3:]))
                if None in noise_q:
                    # print("failed to compute inverse kin")
                    inverse_fails += 1
                    continue
                # To avoid angle wrapping, theta -> sin(theta), cos(theta)
                noise_pos_awrap = np.zeros(9)
                noise_pos_awrap[:3] = noise_pos[:3]
                # print(get_rotation_mat(noise_pos[3:]).flatten('F')[0,:6])
                noise_pos_awrap[3:] = get_rotation_mat(noise_pos[3:]).flatten('F')[0,:6]
                next_pos_awrap = np.zeros(9)
                next_pos_awrap[:3] = next_pos[:3]
                next_pos_awrap[3:] = get_rotation_mat(next_pos[3:]).flatten('F')[0,:6]

                # action = np.zeros(9)
                # action[:3] = next_pos_awrap[:3] - noise_pos_awrap[:3]

                # for i in range(3):
                    # noise_pos_awrap[3+i] = np.sin(noise_pos[3+i])
                    # noise_pos_awrap[6+i] = np.cos(noise_pos[3+i])
                    # next_pos_awrap[3+i] = np.sin(next_pos[3+i])
                    # next_pos_awrap[6+i] = np.cos(next_pos[3+i])
                    
                    # action[3+i] = np.sin(next_pos[3+i]) * np.cos(noise_pos[3+i]) - np.cos(next_pos[3+i]) * np.sin(noise_pos[3+i])
                    # action[6+i] = np.cos(next_pos[3+i]) * np.cos(noise_pos[3+i]) + np.sin(next_pos[3+i]) * np.sin(noise_pos[3+i])
                # print("angles: {}".format(noise_pos_awrap[3:]))
                action = next_pos - noise_pos
                # if not curr_trans_mode[0]:
                # print("pos: {}, ac: {}".format(noise_pos, action))
                # action = next_pos_awrap - noise_pos_awrap
                # action = np.array(demo[idx]["xdot_h"])
                # history = noise_q + noise_pos + curr_gripper_pos + trans_mode + slow_mode
                history = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                state = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                # print(noise_q)
                dataset.append((history, state, z, action.tolist()))
        # sys.exit()
    if inverse_fails > 0:
        print("Failed inverses: {}".format(inverse_fails))
        # sys.exit()
    pickle.dump(dataset, open(savename, "wb"))
    print(dataset[0])
    print("[*] I have this many subtrajectories: ", len(dataset))

    model = CAE().to(device)
    dataname = 'data/' + 'cae.pkl'
    savename = 'models/' + 'cae'

    EPOCH = 50
    LR = 0.0001
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.15

    train_data = MotionData(dataname)
    BATCH_SIZE_TRAIN = 2#int(train_data.__len__() / 10.)
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
    rospy.init_node("train_cae")
    train_cae()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
