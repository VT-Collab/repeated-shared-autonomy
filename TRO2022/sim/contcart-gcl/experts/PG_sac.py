import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .PG import PG


class ModifiedQNetwork(nn.Module):


    # note that it is assumes that 
    # linearList[-2] is the mean layer
    # linearList[-1] is the std layer
    def __init__(self, linearList: nn.ModuleList):
        super(ModifiedQNetwork, self).__init__()
        self.linearList = linearList

        # Q1 architecture
    def forward(self, state):
        x1 = state
        nSequentialLayers = len(self.linearList) - 2
        for i in range(nSequentialLayers):
            x1 = F.relu(self.linearList[i](x1))
        xmean = self.linearList[-2](x1)
        xstd = self.linearList[-1](x1)

        return xmean, xstd

class GaussianPG(PG):

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def __init__(self, state_shape, n_actions, hidden_dim=128, action_space=None):
        
        super(PG, self).__init__()

        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20
        self.epsilon = 1e-6
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.output_space = n_actions

        self.linear1 = nn.Linear(state_shape[0], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear = nn.Linear(hidden_dim, n_actions)
        self.apply(self.weights_init_)
        moduleList = nn.ModuleList([self.linear1, self.linear2, self.mean_linear, self.log_std_linear])
        self.model = ModifiedQNetwork(moduleList)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPG, self).to(device)

    def generate_session(self, env, t_max=1000):
        states, traj_probs, actions, rewards = [], [], [], []
        s = env.reset()
        q_t = 1.0
        oldGym = True
        if not oldGym:
            s = s[0]
        for _ in range(t_max):
            a, _, _ = self.sample(torch.FloatTensor(s))
            a = a.detach()
            # a = torch.cat(self.model(torch.FloatTensor(s))).detach()
            if oldGym:
                a = list(a)

                new_s, r, done, _ = env.step(a)
            else:
                new_s, r, done, _, _ = env.step(a)
            states.append(s)
            # traj_probs.append(q_t)
            actions.append(a)
            rewards.append(r)
            s = new_s
            if done:
                break
        return states, actions, rewards

    def train_on_env(self, env, gamma=0.99, entropy_coef=1e-2):
        states, actions, rewards = self.generate_session(env)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        cumulative_returns = np.array(self._get_cumulative_rewards(rewards, gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

        # logits = self.model(states)
        # since we are sampling from a normal distribution (a PDF), 
        # the probability of an action "logits" being "chosen"
        # is zero 
        #
        # However, the CONFIDENCE can be treated as inversely proportional to the 
        # standard deviation 
        #

        logits, logits_mean, logits_log_stdev = self.sample(states)
        conf = torch.reciprocal(logits_log_stdev)
        print(conf)
        print(logits_log_stdev)
        input()
        probs = torch.exp(conf)
        log_probs = conf

        # log_probs_for_actions = torch.sum(
        #     log_probs * self._to_one_hot(actions, self.output_space), dim=1)
        log_probs_for_actions = log_probs
        print(cumulative_returns)

        entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
        loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*entropy_coef)
        print(loss)
        input()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.sum(rewards)
