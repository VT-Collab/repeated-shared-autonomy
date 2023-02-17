import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from utils import soft_update, hard_update
from model_sac import QNetwork, GaussianPolicy



class SAC(object):
    def __init__(self, action_space, state_dim=12, action_dim=3):

        # hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.0003
        self.hidden_size = 256
        self.target_update_interval = 1
        self.state_dim = state_dim
        self.action_dim = action_dim

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


    def update_parameters(self, memory, batch_size):

        # Sample a batch of data
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
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
            'policy_updates': self.policy_updates,
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict()
        }
        torch.save(checkpoint, "models/{}/sac_{}_{}.pt".format(algo, algo, name))


    def load_model(self, algo, name):

        print('[*] Loading from models/{}/sac_{}_{}.pt'.format(algo, algo, name))

        checkpoint = torch.load("models/{}/sac_{}_{}.pt".format(algo, algo, name))
        self.policy_updates = checkpoint['policy_updates']
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
