import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle
import random
import numpy as np
from sklearn.utils import shuffle
import sys
import os, rospy
from utils import TrajectoryClient
import copy 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glob import glob
from geometry_msgs.msg import Twist
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

np.set_printoptions(precision=2, suppress=True)
# collect dataset
class MotionData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.target = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item)
        label = torch.LongTensor(self.target[idx])
        # print(idx, self.target[idx]), label
        # print(snippet, label)
        return (snippet, label)


# Discriminator
class Net(nn.Module):

    def __init__(self, d_layers=[30, 40, 20, 10], d_in=15, d_out=2):
        super(Net, self).__init__()

        # self.loss_func = nn.CrossEntropyLoss(weight = torch.Tensor([1., 2.]))
        self.loss_func = nn.CrossEntropyLoss()

        fig, self.axs = plt.subplots(3, 2)
        self.axs = self.axs.flat

        # construct FCN
        self.fcn = nn.ModuleList()
        # input layer
        self.fcn.append(nn.Linear(d_in, d_layers[0]))
        # hidden layers
        n_layers = len(d_layers)
        for i in range(n_layers-1):
            layer = nn.Linear(d_layers[i], d_layers[i + 1])
            self.fcn.append(layer)
        # output layer
        self.fcn.append(nn.Linear(d_layers[-1], d_out))

        # output data for plotting
        self.plot_data = []
        
        self.pca = PCA(n_components = 2)

    def classify(self, x):
        self.layer_outputs = []
        out = x
        for i, l in enumerate(self.fcn):
            out = l(out)
            if i == len(self.fcn) - 1:
                self.y_pred = out
                break
            self.layer_outputs.append(out.detach().numpy())
            out = F.tanh(out)
            self.layer_outputs.append(out.detach().numpy())
        
        return self.y_pred

    def plot(self):
        # plotting for debug
        for i, output in enumerate(self.layer_outputs):
            plt_data = self.pca.fit_transform(output)
            plt_data[:,0] = plt_data[:,0]/max(plt_data[:,0])
            plt_data[:,1] = plt_data[:,1]/max(plt_data[:,1])
            self.axs[i].clear()
            self.axs[i].set_xlim([-1.5, 1.5])
            self.axs[i].set_ylim([-1.5, 1.5])
            self.axs[i].plot(plt_data[np.where(self.y_true==0),0],plt_data[np.where(self.y_true==0),1], 'bx')
            self.axs[i].plot(plt_data[np.where(self.y_true==1),0],plt_data[np.where(self.y_true==1),1], 'rx')
        plt.draw()
        plt.pause(0.0001)

    def forward(self, x):
        c = x[0]
        self.y_true = x[1].flatten()
        # print(self.y_true)
        y_output = self.classify(c)
        # self.plot()
        loss = self.loss(y_output, self.y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)


def deform(xi, start, length, tau):
    # length += np.random.choice(np.arange(30, length))
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
def train_classifier():
    mover = TrajectoryClient()
    dataset = []
    parent_folder = 'demos'
    
    deformed_trajs = []
    # noisesamples = 3
    z = [1.0]

    plt.ion()
    plt.show()

    true_cnt = 0
    false_cnt = 0
    savename = 'data/' + 'class.pkl'
    folders = ["place", "pour", "stir"]
    for folder in folders:
        demos = glob(parent_folder + "/" + folder + "/*.pkl")
        for filename in demos:
            demo = pickle.load(open(filename, "rb"))

            # Human's demonstrations
            traj = []
            traj_pos = []
            for item in demo:
                home_pos = np.array(item["start_pos"])
                home_q = np.array(item["start_q"])
                home_gripper_pos = [item["start_gripper_pos"]]
                
                curr_pos = np.asarray(item["curr_pos"])
                curr_q = np.asarray(item["curr_q"])
                curr_gripper_pos = [item["curr_gripper_pos"]]

                curr_trans_mode = [float(item["trans_mode"])]
                curr_slow_mode = [float(item["slow_mode"])]

                traj.append(curr_q.tolist())
                traj_pos.append(curr_pos.tolist())
                curr_pos_awrap = np.zeros(9)
                curr_pos_awrap[:3] = curr_pos[:3]
                curr_pos_awrap[3:] = get_rotation_mat(curr_pos[3:]).flatten('F')[0,:6]
                # curr
                state = curr_q.tolist() + curr_pos_awrap.tolist()
                dataset.append([state, [0]])

            deformed_samples = 2
            tau = np.random.uniform([-0.1]*6 + [0.1]*6)
            deform_len = len(traj)
            traj = np.array(traj)
            # print(np.array(traj_pos)[:,:3])
            for i in range(deformed_samples):
                start = np.random.choice(np.arange(10, int(len(traj)*0.35)))
                # start = 30
                snip_deformed = deform(traj, start, deform_len, tau)
                # snip_deformed = deform(traj[:,0:3], start, deform_len, tau)
                snip_deformed_cart = traj.copy()
                for snip_idx in range(len(snip_deformed)):
                    
                    snip_deformed_cart[snip_idx] = mover.joint2pose(snip_deformed[snip_idx])
                    curr_pos_awrap = np.zeros(9)
                    curr_pos_awrap[:3] = snip_deformed_cart[snip_idx][:3]
                    curr_pos_awrap[3:] = get_rotation_mat(snip_deformed_cart[snip_idx][3:]).flatten('F')[0,:6]
                    state = snip_deformed[snip_idx].tolist() + curr_pos_awrap.tolist()
                    dataset.append([state, [1.]])
                # deformed_trajs.append(snip_deformed_cart)
                # start = np.random.choice(np.arange(0, int(len(traj)*0.35)))
                start = 0
                # snip_deformed = deform(traj[:,0:3], start, deform_len, tau)
                snip_deformed_cart = np.array(traj_pos)

                # tau = np.random.uniform([-0.05]*6, [0.05]*6)
                tau = np.random.uniform([-0.05, -0.02, -0.05, -0.05, -0.05, -0.05], [0.0, 0.05, 0.05, 0.05, 0.05, 0.05])
                snip_deformed = deform(traj_pos, start, deform_len, tau)
                snip_plot = np.array(traj_pos)[:1, :]
                for snip_idx in range(len(snip_deformed)):
                    pos_twist = Twist()
                    pos_twist.linear.x = snip_deformed[snip_idx, 0]
                    pos_twist.linear.y = snip_deformed[snip_idx, 1]
                    pos_twist.linear.z = snip_deformed[snip_idx, 2]
                    pos_twist.angular.x = snip_deformed[snip_idx, 3]
                    pos_twist.angular.y = snip_deformed[snip_idx, 4]
                    pos_twist.angular.z = snip_deformed[snip_idx, 5]
                    snip_deformed_joint = mover.pose2joint(pos_twist, guess=traj[snip_idx])
                    if snip_deformed_joint is None:
                        # snip_deformed_cart[snip_idx] = np.zeros(6)
                        # snip_plot = np.append(snip_plot, snip_deformed_cart[snip_idx].reshape(1,6), axis=0)
                        continue
                    snip_deformed_cart[snip_idx] = snip_deformed[snip_idx]
                    # print("{} {}".format(np.array(traj_pos)[snip_idx], snip_deformed_cart[snip_idx]))
                    snip_plot = np.append(snip_plot, snip_deformed_cart[snip_idx].reshape(1,6), axis=0)
                    curr_pos_awrap = np.zeros(9)
                    curr_pos_awrap[:3] = snip_deformed_cart[snip_idx][:3]
                    curr_pos_awrap[3:] = get_rotation_mat(snip_deformed_cart[snip_idx][3:]).flatten('F')[0,:6]
                    state = snip_deformed_joint.tolist() + curr_pos_awrap.tolist()
                    dataset.append([state, [1.]])
                    # print(pos)
                    # print(np.where(snip_plot[:,:3] < -1.))
                #     print("--")
                # sys.exit()
                # print(snip_plot[:,:3])
                deformed_trajs.append(snip_plot)

    pickle.dump(deformed_trajs, open("deformed_trajs.pkl", "wb"))
    # sys.exit()
        # print(snip_deformed_cart)
        # sys.exit()
            # data = [item[0][6:] for item in demo]
            # traj = torch.Tensor(traj).unsqueeze(0)
            # label = np.zeros(len(demo)).tolist()
            # # label = 0.
            # dataset.append([traj, label])
    # print(dataset)
    # print(len(dataset))
    pickle.dump(dataset, open(savename, "wb"))
    # sys.exit()
    model = Net().to(device)
    dataname = 'data/' + 'class.pkl'
    savename = 'models/' + 'class'

    EPOCH = 45
    # BATCH_SIZE_TRAIN = 10000
    LR = 0.0001
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1


    raw_data = pickle.load(open(dataname, "rb"))
    raw_data = random.sample(raw_data, len(raw_data))
    # print(raw_data)
    # raw_data = raw_data.tolist()
    inputs = [element[0] for element in raw_data]
    targets = [element[1] for element in raw_data]
    # print(len(inputs), len(targets))
    # print(targets)

    train_data = MotionData(inputs, targets)
    # BATCH_SIZE_TRAIN = int(train_data.__len__() / 5.)
    BATCH_SIZE_TRAIN = 1#int(train_data.__len__() / 5.)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    plot_data = []
    
    for epoch in range(EPOCH):
        # deformed_trajs = []
        for batch, x in enumerate(train_set):
            # network_inputs = []
            # network_labels = []
            # for dataset in x:

            #     traj = x[0][0].numpy()
            #     labels = x[1][0].numpy()

            #     network_inputs += traj.tolist()
            #     network_labels += labels.tolist()

            #     deformed_samples = 2
            #     tau = np.random.uniform([-0.1]*6, [0.1]*6)
            #     deform_len = len(traj)
            #     for i in range(deformed_samples):
            #         start = np.random.choice(np.arange(0, int(len(traj)*0.5)))
            #         # start = 5
            #         snip_deformed = deform(traj[:,6:12], start, deform_len, tau)
            #         # snip_deformed = deform(traj[:,0:3], start, deform_len, tau)
            #         deformed_trajs.append(snip_deformed)
            #         traj[:,6:12] = snip_deformed
            #         # traj[:,0:3] = snip_deformed
            #         labels = np.ones(deform_len)
            #         network_inputs += traj.tolist()
            #         network_labels += labels.tolist()
            # network_inputs = torch.FloatTensor(network_inputs)
            # network_labels = torch.LongTensor(network_labels)
            # r_idx = torch.randperm(len(network_inputs))
            # network_inputs = network_inputs[r_idx, :]
            # network_labels = network_labels[r_idx]
            # x = [network_inputs, network_labels]
            optimizer.zero_grad()
            loss = model(x)
            # save data for plotting
            if epoch % 50 == 0:
                data_dict = {}
                data_dict["epoch"] = epoch
                data_dict["batch"] = batch
                data_dict["data"] = model.layer_outputs
                data_dict["gt"] = model.y_true
                data_dict["pred"] = model.y_pred
                plot_data.append(data_dict)
            loss.backward()
            optimizer.step()
        # if not epoch:
        #     pickle.dump(deformed_trajs, open("deformed_trajs.pkl", "wb"))
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)
    pickle.dump(plot_data, open("plot_data.pkl", "wb"))
    

def main():
    rospy.init_node("train_class")
    train_classifier()

if __name__ == "__main__":
    main()
