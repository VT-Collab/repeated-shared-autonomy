import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle, random, argparse
import numpy as np
import sys, rospy
from utils import TrajectoryClient, convert_to_6d, deform
from glob import glob
from geometry_msgs.msg import Twist

device = "cpu"

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
        return (snippet, label)

class Net(nn.Module):

    def __init__(self, d_layers=[30, 40, 20, 10], d_in=15, d_out=2):
        super(Net, self).__init__()

        # self.loss_func = nn.CrossEntropyLoss(weight = torch.Tensor([1., 2.]))
        self.loss_func = nn.CrossEntropyLoss()

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

    def classify(self, x):
        # save outputs from individual layers
        self.layer_outputs = []
        out = x
        for i, l in enumerate(self.fcn):
            out = l(out)
            if i == len(self.fcn) - 1:
                self.y_pred = out
                break
            self.layer_outputs.append(out.detach().numpy())
            out = torch.tanh(out)
            self.layer_outputs.append(out.detach().numpy())
        
        return self.y_pred

    def forward(self, x):
        c = x[0]
        self.y_true = x[1].flatten()
        y_output = self.classify(c)
        loss = self.loss(y_output, self.y_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)

def train_classifier(args):
    mover = TrajectoryClient()

    parent_folder = 'demos'
    folders = ["place", "pour", "stir"]
    # only pick requested tasks from list
    folders = folders[:args.n_tasks]
    rospy.loginfo("Using demos for tasks : {}".format(folders))

    data_folder = 'data'
    model_folder = 'models'
    savename = 'class_' + "_".join(folders)
    
    true_cnt = 0
    false_cnt = 0
    dataset = []
    deformed_trajs = []

    for folder in folders:
        rospy.loginfo("Generating deformations for task : {}".format(folder))
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

                # euler angles is insufficient for nn training. Get 6d representation. 
                # See https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
                curr_pos_awrap = convert_to_6d(curr_pos)
                state = curr_q.tolist() + curr_pos_awrap.tolist()
                # class 0 for real states
                dataset.append([state, [0]])
                true_cnt += 1

            traj = np.array(traj)
            traj_pos = np.array(traj_pos)

            for _ in range(args.deforms):
                # # joint based deformations
                # deform_len = len(traj)
                # start = np.random.choice(np.arange(10, int(len(traj)*0.35)))
                # # force along each joint
                # tau = np.random.uniform([-0.05]*6 + [0.05]*6)

                # snip_deformed = deform(traj, start, deform_len, tau)
                # snip_deformed_cart = traj.copy()

                # # compute fk for each joint pose
                # for snip_idx in range(len(snip_deformed)):
                #     snip_deformed_cart[snip_idx] = mover.joint2pose(snip_deformed[snip_idx])
                #     curr_pos_awrap = convert_to_6d(snip_deformed_cart[snip_idx])
                #     state = snip_deformed[snip_idx].tolist() + curr_pos_awrap.tolist()
                #     dataset.append([state, [1.]])
                #     false_cnt += 1
                # deformed_trajs.append(snip_deformed)

                # cartesian position based deformations
                deform_len = len(traj)
                start = 0
                tau = np.random.uniform([-0.05, -0.02, -0.05, -0.05, -0.05, -0.05], [0.0, 0.05, 0.05, 0.05, 0.05, 0.05])

                snip_deformed = deform(traj_pos, start, deform_len, tau)
                snip_deformed_cart = traj_pos.copy()
                # save cartesian positions for plotting
                snip_plot = np.array(traj_pos)[:1, :]

                for snip_idx in range(len(snip_deformed)):
                    # convert to twist msg for kdl_kin
                    pos_twist = Twist()
                    pos_twist.linear.x = snip_deformed[snip_idx, 0]
                    pos_twist.linear.y = snip_deformed[snip_idx, 1]
                    pos_twist.linear.z = snip_deformed[snip_idx, 2]
                    pos_twist.angular.x = snip_deformed[snip_idx, 3]
                    pos_twist.angular.y = snip_deformed[snip_idx, 4]
                    pos_twist.angular.z = snip_deformed[snip_idx, 5]

                    snip_deformed_joint = mover.pose2joint(pos_twist, guess=traj[snip_idx])
                    # valid inverse not found. Ignore waypoint
                    if snip_deformed_joint is None:
                        continue

                    snip_deformed_cart[snip_idx] = snip_deformed[snip_idx]
                    snip_plot = np.append(snip_plot, snip_deformed_cart[snip_idx].reshape(1,6), axis=0)
                    curr_pos_awrap = convert_to_6d(snip_deformed_cart[snip_idx])
                    state = snip_deformed_joint.tolist() + curr_pos_awrap.tolist()
                    dataset.append([state, [1.]])
                    false_cnt += 1
                    
                deformed_trajs.append(snip_plot)

    rospy.loginfo("Real waypoints: {} deformations: {}".format(true_cnt, false_cnt))
    # save deformations for plotting
    pickle.dump(deformed_trajs, open(data_folder + "/" +"deformed_trajs.pkl", "wb"))
    pickle.dump(dataset, open(data_folder + "/" + savename + ".pkl", "wb"))
    
    model = Net().to(device)

    # Shuffle dataset
    dataset = random.sample(dataset, len(dataset))
    inputs = [element[0] for element in dataset]
    targets = [element[1] for element in dataset]

    # Training parameters
    EPOCH = 45
    # BATCH_SIZE_TRAIN = int(train_data.__len__() / 5.)
    BATCH_SIZE_TRAIN = 1
    LR = 0.0001
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.1

    train_data = MotionData(inputs, targets)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    plot_data = []

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):

            optimizer.zero_grad()
            loss = model(x)

            # save data for plotting
            if epoch % 10 == 0:
                data_dict = {}
                data_dict["epoch"] = epoch
                data_dict["batch"] = batch
                data_dict["data"] = model.layer_outputs
                data_dict["gt"] = model.y_true
                data_dict["pred"] = model.y_pred
                plot_data.append(data_dict)

            loss.backward()
            optimizer.step()

        scheduler.step()
        rospy.loginfo("epoch: {} loss: {}".format(epoch, loss.item()))
        torch.save(model.state_dict(), model_folder + "/" + savename)

    pickle.dump(plot_data, open(data_folder + "/" + "plot_data.pkl", "wb"))
    

def main():
    rospy.init_node("train_class")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, help="number of tasks to use", default=1)
    parser.add_argument("--deforms", type=int, help="number of deformations per demo", default=1)
    args = parser.parse_args()
    train_classifier(args)

if __name__ == "__main__":
    main()
