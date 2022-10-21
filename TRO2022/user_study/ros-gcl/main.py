import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from cost import CostNN
from utils import to_one_hot, get_cumulative_rewards
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
from utils import TrajectoryClient, convert_to_6d
from geometry_msgs.msg import Twist
from experts.PG import PG

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

def record_traj():
    state = None
    action = None

    return state, action 

# learns an intent using GCL given expert demonstrations from an intentFolder
# saves the policy to intents/{intentFolder}/ and cost to intents/{intentFolder}/cost
def learn_intent(args):
    robot_interface = None
    intentFolder = "intent" + str(args.intentFolder)
    # SEEDS
    seed = 1666219353 # unnecessary
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # LOADING EXPERT/DEMO SAMPLES
    # demo_trajs = np.load('expert_samples/pg_contcartpole.npy', allow_pickle=True)
    if args.generate:
        generate_dataset(args)
    demo_trajs = pickle.load("intent_samples/"+intentFolder+"/ALL")
    state_shape = len(demo_trajs[0][0]) # demo trajs = (state, action), size of the state space
    n_actions = len(demo_trajs[0][1]) # size of the action space
    init_state = None
    state = init_state # needs to be measured from environment TODO

    input()
    print(len(demo_trajs))

    # INITILIZING POLICY AND REWARD FUNCTION
    policy = PG(state_shape, n_actions)
    cost_f = CostNN(state_shape[0] + 1)
    policy_optimizer = torch.optim.Adam(policy.parameters(), 1e-2)
    cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)

    mean_rewards = []
    mean_costs = []
    mean_loss_rew = []
    EPISODES_TO_PLAY = 1
    REWARD_FUNCTION_UPDATE = 10
    DEMO_BATCH = 1000
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
            states = np.array(traj[0])

            if is_Demo:
                # then probs are certain
                probs = np.ones((states.shape[0], 1))
            else:
                # probs are calculated in generate_real_session
                probs = np.array(traj[3]).reshape(-1, 1)

            # note that actions is the input given to the controller, 
            # not the index of the action in the lookup table
            actions = np.array(traj[1]).reshape(-1, 1)
            x = np.concatenate((states, probs, actions), axis=1)
            step_list.extend(x)
        return np.array(step_list)

    D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
    return_list, sum_of_cost_list = [], []
    for i in range(1000):
        trajs = [policy.generate_real_session(robot_interface) for _ in range(EPISODES_TO_PLAY)]
        sample_trajs = trajs + sample_trajs
        D_samp = preprocess_traj(trajs, D_samp)

        # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
        loss_rew = []
        for _ in range(REWARD_FUNCTION_UPDATE):
            selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
            selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

            D_s_samp = D_samp[selected_samp]
            D_s_demo = D_demo[selected_demo]

            D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0)

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

            # LOSS CALCULATION FOR IOC (COST FUNCTION)
            loss_IOC = torch.mean(costs_demo) + \
                    torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
            # UPDATING THE COST FUNCTION
            cost_optimizer.zero_grad()
            loss_IOC.backward()
            cost_optimizer.step()

            loss_rew.append(loss_IOC.detach())

        for traj in trajs:
            states, actions, rewards = traj

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)

            costs = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()
            cumulative_returns = np.array(get_cumulative_rewards(-costs, 0.99))
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

        returns = sum(rewards)
        sum_of_cost = np.sum(costs)
        return_list.append(returns)
        sum_of_cost_list.append(sum_of_cost)

        mean_rewards.append(np.mean(return_list))
        mean_costs.append(np.mean(sum_of_cost_list))
        mean_loss_rew.append(np.mean(loss_rew))

        # PLOTTING PERFORMANCE
        # if i % 10 == 0:
        #     pass
            # # clear_output(True)
            # print(f"mean reward:{np.mean(return_list)} loss: {loss_IOC}")

            # plt.figure(figsize=[16, 12])
            # plt.subplot(2, 2, 1)
            # plt.title(f"Mean reward per {EPISODES_TO_PLAY} games")
            # plt.plot(mean_rewards)
            # plt.grid()

            # plt.subplot(2, 2, 2)
            # plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
            # plt.plot(mean_costs)
            # plt.grid()

            # plt.subplot(2, 2, 3)
            # plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
            # plt.plot(mean_loss_rew)
            # plt.grid()

            # # plt.show()
            # plt.savefig('plots/GCL_learning_curve.png')
            # plt.close()

        if np.mean(return_list) > 500:
            break



if __name__ == "__main__":
    rospy.init_node("train_intent")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, help="number of tasks to use", default=1)
    parser.add_argument("--lookahead", type=int, help="lookahead to compute robot action", default=5)
    parser.add_argument("--noisesamples", type=int, help="num of noise samples", default=5)
    parser.add_argument("--noiselevel", type=float, help="variance for noise", default=.0005)
    parser.add_argument("--intentFolder", type=int, help="intent folder to read from and save to", default=0)
    parser.add_argument("--generate", type=bool, help="whether to (re)generate the intent dataset for intentFolder", default=False)
    args = parser.parse_args()
    learn_intent(args)