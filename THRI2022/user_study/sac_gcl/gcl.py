import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from utils import soft_update, hard_update
from model_sac import QNetwork, GaussianPolicy
from model_gcl import CostNN



class GCL(object):
    def __init__(self, action_space=None, state_dim=12, action_dim=9):
        
        # hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.0003
        self.hidden_size = 128
        self.target_update_interval = 1
        self.state_dim = state_dim
        self.action_dim = action_dim

        if action_space is None:
            action_space = {"high":np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0]), 
                            "low":-np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0])}

        # Cost
        self.cost_f = CostNN(self.state_dim, hidden_dim=128)
        self.cost_optim = Adam(self.cost_f.parameters(), lr=0.001)
        self.cost_updates = 0

        # Critic
        self.critic = QNetwork(num_inputs=self.state_dim, num_actions=self.action_dim, hidden_dim=self.hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(num_inputs=self.state_dim, num_actions=self.action_dim, hidden_dim=self.hidden_size)
        hard_update(self.critic_target, self.critic)

        # Actor
        self.policy = GaussianPolicy(num_inputs=self.state_dim, num_actions=self.action_dim, hidden_dim=self.hidden_size, action_space=action_space)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.policy_updates = 0


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.detach().numpy()[0]


    def update_cost(self, memory_expert, memory_novice, batch_size):

        # sample states from expert and novice
        state_batch_expert, _, _, _, _ = memory_expert.sample(batch_size=batch_size)
        state_batch_novice, _, _, _, _ = memory_novice.sample(batch_size=batch_size, flag=True)
        state_batch_expert = torch.FloatTensor(state_batch_expert)
        state_batch_novice = torch.FloatTensor(state_batch_novice)

        # append the demonstration batch to the novice batch
        state_batch_novice = torch.cat((state_batch_expert, state_batch_novice), 0)

        # calculate reward
        costs_demo = self.cost_f(state_batch_expert)
        costs_samp = self.cost_f(state_batch_novice)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)))

        # update cost function
        self.cost_optim.zero_grad()
        loss_IOC.backward()
        self.cost_optim.step()
        self.cost_updates += 1

        return loss_IOC.item()


    def update_policy(self, memory_novice, batch_size):

        # Sample a batch of data
        state_batch, action_batch, _, next_state_batch, mask_batch = memory_novice.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = -self.cost_f(state_batch).detach()
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1)

        # train critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # train actor
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.policy_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        self.policy_updates += 1

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()


    def save_model(self, algo, name):

        print('[*] Saving as models/{}/sac_{}_{}.pt'.format(algo, algo, name))
        if not os.path.exists('models/{}/'.format(algo)):
            os.makedirs('models/{}/'.format(algo))

        checkpoint = {
            'cost_updates': self.cost_updates,
            'policy_updates': self.policy_updates,
            'cost': self.cost_f.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'cost_optim': self.cost_optim.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict()
        }
        torch.save(checkpoint, "models/{}/sac_{}_{}.pt".format(algo, algo, name))


    def load_model(self, algo, name):

        print('[*] Loading from models/{}/sac_{}_{}.pt'.format(algo, algo, name))

        checkpoint = torch.load("models/{}/sac_{}_{}.pt".format(algo, algo, name))
        self.cost_updates = checkpoint['cost_updates']
        self.policy_updates = checkpoint['policy_updates']
        self.cost_f.load_state_dict(checkpoint['cost'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])
        self.cost_optim.load_state_dict(checkpoint['cost_optim'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])

    def load_model_filename(self, name, algo="gcl"):

        print('[*] Loading from models/{}/{}'.format(algo, name))
        # hardcode checkpoint location
        folder = "/home/ur10/ur10_ws/src/repeated-shared-autonomy/THRI2022/user_study/sac_gcl/"
        checkpoint = torch.load(folder + "models/{}/{}".format(algo, name))
        self.cost_updates = checkpoint['cost_updates']
        self.policy_updates = checkpoint['policy_updates']
        self.cost_f.load_state_dict(checkpoint['cost'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])
        self.cost_optim.load_state_dict(checkpoint['cost_optim'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
