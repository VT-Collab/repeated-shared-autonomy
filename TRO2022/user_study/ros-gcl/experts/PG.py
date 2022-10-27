import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import rospy
from scipy import stats

class PG(nn.Module):

    def generate_lookup_table(self, action_space_dim, values=[-1, 0, 1]):
        return np.array(list(product(values, repeat=action_space_dim)))
    
    def temp_table(self):
        return [[0, 0, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 0, -1, 1, 0, 1]]

    def one_trans_per_step(self):
        return [[0, 0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 1],
                [-1, 0, 0, 1, 0, 1],
                [0, -1, 0, 1, 0, 1],
                [0, 0, -1, 1, 0, 1],]

    def generate_robot_action_table(self, axisValues=[-1, 0, 1]):
        a = []
        possibleAxes = list(product(axisValues, repeat=3))
        # possibleLogics = [[0, 0, 1], [1, 0, 1]]
        possibleLogics = [[1, 0, 1]]
        # possibleLogics = list(product([0], repeat=3))
        for z in possibleAxes:
            for y in possibleLogics:
                a.append(list(z) + list(y))
        return np.array(a)

    def __init__(self, state_shape, n_actions, v = [-1, 0, 1]):
        super(PG, self).__init__()
        self.state_shape = state_shape
        # self.action_table = self.generate_robot_action_table(axisValues=v)
        self.action_table = self.one_trans_per_step()
        self.n_actions = n_actions
        self.output_space = len(self.action_table)
        self.model = nn.Sequential(
            nn.Linear(in_features = state_shape[0], out_features = 128),
            nn.ReLU(),
            # nn.Linear(in_features = 64 , out_features = self.output_space)
            nn.Linear(in_features = 128 , out_features = 64), 
            nn.ReLU(),
            nn.Linear(in_features = 64 , out_features = self.output_space)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
    def forward(self, x):
        logits = self.model(x)
        return logits

    
    def predict_probs(self, states):
        states = torch.FloatTensor(states)
        logits = self.model(states).detach()
        probs = F.softmax(logits, dim = -1).numpy()
        # print(states, logits, probs)
        return probs
    
    def generate_session(self, env, t_max=100):
        states, traj_probs, actions, rewards = [], [], [], []
        s = env.reset()
        q_t = 1.0
        oldGym = False
        if not oldGym:
            s = s[0]

        for t in range(t_max):
            action_probs = self.predict_probs(np.array([s]))[0]
            a_choice = np.random.choice(self.output_space,  p = action_probs)
            a = self.action_table[a_choice]
            print(a_choice)
            if oldGym:
                # need to change action to a float
                a = list(a)
                new_s, r, done, _ = env.step(a)
            # note that a is an array
            else:
                new_s, r, done, info, _ = env.step(a)
            q_t *= action_probs[a_choice]

            states.append(s)
            traj_probs.append(q_t)
            actions.append(a_choice)
            rewards.append(r)

            s = new_s
            if done:
                break
        return states, actions, rewards, traj_probs 

    def generate_real_session(self, robot_interface, t_max=1000):
        states, traj_probs, actions = [], [], []
        s = robot_interface.getState()
        q_t = 1.0
        stalled = False
        done = False
        for t in range(t_max):

            if done and not stalled: # outside of boundaries
                stalled = True
                # for some reason this wasn't working well
                STALL_LEN = 300 # should be sufficient
                for i in range(STALL_LEN):
                    robot_interface.applyAction([0, 0, 0, 1, 0, 1])

            action_probs = self.predict_probs(np.array([s]))[0]
            a_choice = np.random.choice(self.output_space,  p = action_probs)
            a = self.action_table[a_choice]
            if not stalled:
                new_s, done = robot_interface.applyAction(a)
            # if stalled:

            #     robot_interface.applyAction([0, 0, 0, 1, 0, 1])
            
            q_t *= action_probs[a_choice]
            states.append(s)
            traj_probs.append(q_t)
            actions.append(a_choice)

            s = new_s
        robot_interface.goHome()
        return states, actions, traj_probs 

    def _get_cumulative_rewards(self, rewards, gamma=0.99):
        G = np.zeros_like(rewards, dtype = float)
        G[-1] = rewards[-1]
        for idx in range(-2, -len(rewards)-1, -1):
            G[idx] = rewards[idx] + gamma * G[idx+1]
        return G

    def _to_one_hot(self, y_tensor, ndims):
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(
            y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
        return y_one_hot

    def train_on_env(self, env, gamma=0.99, entropy_coef=1e-2):
        states, actions, rewards = self.generate_session(env)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        cumulative_returns = np.array(self._get_cumulative_rewards(rewards, gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        log_probs_for_actions = torch.sum(
            log_probs * self._to_one_hot(actions, self.output_space), dim=1)
    
        entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
        loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*entropy_coef)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.sum(rewards)

    def train_on_real(self, trainData, rewards, gamma=0.99, entropy_coef=1e-2):
        states, actions = trainData
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        cumulative_returns = np.array(self._get_cumulative_rewards(rewards, gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        log_probs_for_actions = torch.sum(
            log_probs * self._to_one_hot(actions, self.output_space), dim=1)
    
        entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
        loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*entropy_coef)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.sum(rewards), loss.item()        
        