import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from cost import CostNN
from utils import to_one_hot, get_cumulative_rewards, TrajectoryClient, convert_to_6d
from torch.optim.lr_scheduler import StepLR
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import rospy, argparse
from glob import glob
from geometry_msgs.msg import Twist
from experts.PG import PG
from waypoints import HOME
import time 
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D

class RobotInterface():
    def __init__(self):
        self.rate = rospy.Rate(100)
        self.mover = TrajectoryClient()
        self.home_pose = self.mover.joint2pose(HOME)
        self.home_xyz = self.home_pose[0:3]
        rospy.loginfo("Initialized, Moving Home")
        self.mover.go_home()
        self.mover.reset_gripper()

    def goHome(self):
        x = self.mover.go_home()
        self.mover.reset_gripper()
        return x
    # action space is expected to be a dict with the following:
    # axes: (3,)
    # trans_mode, slow_mode, gripper_ac
    def applyAction(self, action):
        if type(action) == dict:
            axes = action["axes"]
            gripper = action["gripper_ac"]
            trans_mode = action["trans_mode"]
            slow_mode = action["slow_mode"]
        else: 
            axes = action[0:3]
            trans_mode = action[3]
            gripper = action[4]
            slow_mode = action[5]
        gripper_ac = 0
        if gripper and (self.mover.robotiq_joint_state > 0):
            self.mover.actuate_gripper(gripper_ac, 1, 0.) # async?
        
        elif gripper and (self.mover.robotiq_joint_state == 0):
            gripper_ac = 1
            self.mover.actuate_gripper(gripper_ac, 1, 0) # is this async?
 
        else: 
            scaling_trans = 0.2 - 0.1*slow_mode 
            scaling_rot = 0.4 - 0.2*slow_mode 
            
            xdot_h = np.zeros(6)

            if trans_mode: 
                xdot_h[:3] = scaling_trans * np.asarray(axes)

            elif not trans_mode:
                # change coord frame from robotiq gripper to tool flange
                R = np.mat([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
                P = np.array([0, 0, -0.10])
                P_hat = np.mat([[0, -P[2], P[1]],
                                [P[2], 0, -P[0]],
                                [-P[1], P[0], 0]])
                
                axes = np.array(axes)[np.newaxis]
                trans_vel = scaling_rot * P_hat * R * axes.T
                rot_vel = scaling_rot * R * axes.T
                xdot_h[:3] = trans_vel.T[:]
                xdot_h[3:] = rot_vel.T[:]
                
            qdot_h = self.mover.xdot2qdot(xdot_h)
            qdot_h = qdot_h.tolist()
            
            qdot_h = self.mover.compute_limits(qdot_h)
            self.mover.send(qdot_h[0])
        self.rate.sleep()
        return self.getState(), self.isDone() 

    def stall(self):
        qdot_h = self.mover.xdot2qdot(self.getState()[-6:-3])
        self.mover.send(qdot_h.tolist()[0])
        return

    def isDone(self):
        xyz = self.getState()[-6:-3]
        outsideArea = np.linalg.norm(np.subtract(xyz, self.home_xyz)) > (self.home_xyz[-1] - 0.15)
        nearBase = np.linalg.norm(np.subtract(xyz, [.05, 0.2, 0.42])) < 0.15
        closeToWires = xyz[0] > 0.52
        tooHigh = xyz[2] > 0.85
        tooLow = xyz[2] < 0.28
        limitY = xyz[1] < 0.18
        return outsideArea or nearBase or closeToWires or tooHigh or limitY or tooLow


    # observe new states and return
    def getState(self):
        q = self.mover.joint_states
        curr_pos = self.mover.joint2pose()
        return list(q) + list(curr_pos)

    


def generate_dataset(args):
    mover = TrajectoryClient()

    folder = "intent" + str(args.intentFolder)
    parent_folder = "intent_samples/"

    lookahead = args.lookahead#5
    noiselevel = args.noiselevel#0.0005
    noisesamples = args.noisesamples#5
    dataset = []
    demos = glob(parent_folder + folder + "/*.pkl")

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

                if None in noise_q:
                    inverse_fails += 1
                    continue

                # Angle wrapping is a bitch
                noise_pos_awrap = convert_to_6d(noise_pos)
                # next_pos_awrap = convert_to_6d(next_pos)

                action = next_pos - noise_pos 

                # history = noise_q + noise_pos + curr_gripper_pos + trans_mode + slow_mode
                # history = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            # + curr_trans_mode + curr_slow_mode
                state = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                # only need state for PG; do not need history
                dataset.append((state, action.tolist()))

    if inverse_fails > 0:
        rospy.loginfo("Failed inverses: {}".format(inverse_fails))

    pickle.dump(dataset, open( parent_folder + folder + "/ALL", "wb"))
    
    # return dataset # slow 

def learn_intent_args(args):
    return learn_intent(args.intentFolder, args.generate, args)

# learns an intent using GCL given expert demonstrations from an intentFolder
# saves the policy to intents/{intentFolder}/ and cost to intents/{intentFolder}/cost
def learn_intent(inpIntentFolder, inpGen, args=None):
    start_time = str(int(time.time()))
    robot_interface = RobotInterface()
    intentFolder = "intent" + str(inpIntentFolder)
    # SEEDS
    seed = 1666219353 # unnecessary
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # LOADING EXPERT/DEMO SAMPLES
    # demo_trajs = np.load('expert_samples/pg_contcartpole.npy', allow_pickle=True)
    if inpGen:
        generate_dataset(args)
    demo_trajs = pickle.load(open("intent_samples/"+intentFolder+"/all_actions_simple.pkl", "r"))
    state_shape = (len(demo_trajs[0][0]),) # demo trajs = (state, action), size of the state space

    # state_shape should be (12,)
    # n_actions = len(demo_trajs[0][1]) # size of the action space
    n_actions = 6
    init_state = robot_interface.getState()

    # INITILIZING POLICY AND REWARD FUNCTION
    policy = PG(state_shape, n_actions)
    cost_f = CostNN(state_shape[0] + 1)
    policy_optimizer = torch.optim.Adam(policy.parameters(), 1e-3)
    cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-3, weight_decay=1e-4)

    mean_rewards = []
    mean_costs = []
    mean_loss_rew = []
    EPISODES_TO_PLAY = 5
    REWARD_FUNCTION_UPDATE = 30
    DEMO_BATCH = 50 # 50 maybe better?
    sample_trajs = []

    D_demo, D_samp = np.array([]), np.array([])

    # CONVERTS TRAJ LIST TO STEP LIST
    # BAC: I think this function has a bug in it that causes it to only work when there are two discrete
    # options directly due to the way that the "actions" are recorded. Note that the actions are recorded
    # as the second term in the model.generate_session function, but they are read as the THIRD term
    # in this function. The third term of every traj in traj_list is actually the reward! I don't know
    # how the author missed this. I must be missing something huge and obvious

    # bac implementation: traj = (state, action). Prob is calculated in this func
    def preprocess_traj(traj_list, step_list, is_Demo = False, action_table=None):
        step_list = step_list.tolist()
        for traj in traj_list:
            if is_Demo:
                states = np.array([traj[0]])
                # then probs are certain
                probs = np.ones((states.shape[0], 1))
            else:
                states = np.array(traj[0])
                # probs are calculated in generate_real_session
                probs = np.array(traj[2]).reshape(-1, 1)

            # note that actions is the input given to the controller, 
            # not the index of the action in the lookup table
            actions = np.array(traj[1]).reshape(-1, 1)
            x = np.concatenate((states, probs, actions), axis=1)
            step_list.extend(x)
        return np.array(step_list)
    D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)

    xdata = []
    ydata = []
    zdata = []

    for i in range(1000):
        trajs = [policy.generate_real_session(robot_interface, t_max=2000)  for _ in range(EPISODES_TO_PLAY)]
        sample_trajs = trajs + sample_trajs
        D_samp = preprocess_traj(trajs, D_samp)

        # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
        loss_rew = []
        my_costs = [[], []]
        for _ in range(REWARD_FUNCTION_UPDATE):
            selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
            selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

            D_s_samp = D_samp[selected_samp]
            D_s_demo = D_demo[selected_demo]

            # D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0) #why?

            states, probs, actions = D_s_samp[:,:-2], D_s_samp[:,-2], D_s_samp[:,-1]
            states_expert, actions_expert = D_s_demo[:,:-2], D_s_demo[:,-1]

            # Reducing from float64 to float32 for making computaton faster
            states = torch.tensor(states, dtype=torch.float32)
            probs = torch.tensor(probs, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)

            states_expert = torch.tensor(states_expert, dtype=torch.float32)
            actions_expert = torch.tensor(actions_expert, dtype=torch.float32)
            costs_samp = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1))
            costs_demo = cost_f(torch.cat((states_expert, actions_expert.reshape(-1, 1)), dim=-1))
            my_costs[0] += [j[0] for j in costs_demo.detach().numpy()]
            my_costs[1] += [j for j in np.setdiff1d(costs_samp.detach().numpy(), costs_demo.detach().numpy())]
            # LOSS CALCULATION FOR IOC (COST FUNCTION)
            loss_IOC = torch.mean(costs_demo) + \
                    torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
            # UPDATING THE COST FUNCTION
            cost_optimizer.zero_grad()
            loss_IOC.backward()
            cost_optimizer.step()

            loss_rew.append(loss_IOC.detach())


        # plot cost
        fig, ax = plt.subplots()
        ax.boxplot(my_costs)
        plt.savefig("plots/costf_" + str(start_time)+ "_" + str(i) + ".png")
        plt.close()
        # 
        rew = []
        for traj in trajs:
            states, actions, probs = traj
            dist = 0
            for s in states:
                xyz = s[-6:-3]
                dist -= np.linalg.norm(np.subtract(xyz, [robot_interface.home_xyz[0], robot_interface.home_xyz[1], 0.15]))
            rew.append(dist)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)

            costs = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()
            cumulative_returns = np.array(get_cumulative_rewards(-costs, 1.00)) # changed from 0.99
            cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

            logits = policy(states)
            probs = nn.functional.softmax(logits, -1)
            log_probs = nn.functional.log_softmax(logits, -1)

            log_probs_for_actions = torch.sum(
                log_probs * to_one_hot(actions, policy.output_space), dim=1)
          
       
            entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
            loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*1e-2)
    
            # UPDATING THE POLICY NETWORK
            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
        rospy.loginfo("Epoch {}. Reward: {}. Demo cost: {}. Sample cost: {}. Average loss: {}".format(str(i), str(np.mean(rew)), 
            str(np.mean(my_costs[0])), str(round(np.mean(my_costs[1]))), str(round(np.mean(np.mean(loss.item()))))))
        torch.save(cost_f.state_dict(), "intents/"+ intentFolder + "/cost_model_simple" + start_time)
        torch.save(policy.state_dict(), "intents/"+ intentFolder + "/policy_model_simple" + start_time)


        # # want cost per action per iteration
        # augmented_cost = len(policy.action_table) * [0]
        # cy = []
        # numruns = len(costs)
        # for j in range(len(costs)):
        #     action = int(actions[j].item())
        #     cy.append(action)
        #     cost = costs[j].item()
        #     augmented_cost[action] += cost
        # xdata += len(augmented_cost) * [i]
        # ydata += [i for i in range((len(augmented_cost)))]
        # zdata += augmented_cost
    
        # fig = plt.figure(1)
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
        # ax.set_xlabel('iteration')
        # ax.set_ylabel('action')
        # ax.set_zlabel('cost')




        # # plt.show()
        # plt.savefig('plots/costs.png')
        # plt.close()

    # save the final models
    torch.save(cost_f.state_dict(), "intents/"+ intentFolder + "/cost_model_simple")
    torch.save(policy.state_dict(), "intents/"+ intentFolder + "/policy_model_simple")
    # done saving


if __name__ == "__main__":
    rospy.init_node("train_intent")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, help="number of tasks to use", default=1)
    parser.add_argument("--lookahead", type=int, help="lookahead to compute robot action", default=5)
    parser.add_argument("--noisesamples", type=int, help="num of noise samples", default=5)
    parser.add_argument("--noiselevel", type=float, help="variance for noise", default=.0005)
    parser.add_argument("--intentFolder", type=int, help="intent folder to read from and save to", default=4)
    parser.add_argument("--generate", type=bool, help="whether to (re)generate the intent dataset for intentFolder", default=False)
    args = parser.parse_args()
    learn_intent_args(args)