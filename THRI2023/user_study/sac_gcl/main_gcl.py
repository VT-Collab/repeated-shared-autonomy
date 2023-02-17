import argparse
import datetime
import gym
from gcl import GCL
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--expert', default="None")
parser.add_argument('--num_eps', type=int, default=100)
parser.add_argument('--start_eps', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--memory_size', type=int, default=10000)
args = parser.parse_args()


# Environment
env = gym.make("gcl-target-v0")

# Agent
agent = GCL(env.action_space)

# Tensorboard
folder = "runs/gcl/"
writer = SummaryWriter(folder + '{}'.format(datetime.datetime.now().strftime("%m-%d_%H-%M")))

# Memory
memory_novice = ReplayMemory(capacity=args.memory_size)
memory_expert = ReplayMemory(capacity=args.memory_size)
memory_expert.load_buffer("expert", args.expert)

# Main loop
for i_episode in range(1, args.num_eps+1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:

        if i_episode < args.start_eps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # update score function
        if len(memory_novice) > args.batch_size:
            cost_loss = agent.update_cost(memory_expert, memory_novice, args.batch_size)
            writer.add_scalar('cost/loss', cost_loss, agent.cost_updates)

        # update policy
        if len(memory_novice) > args.batch_size:
            critic_1_loss, critic_2_loss, policy_loss = agent.update_policy(memory_novice, args.batch_size)
            writer.add_scalar('SAC/critic_1', critic_1_loss, agent.policy_updates)
            writer.add_scalar('SAC/critic_2', critic_2_loss, agent.policy_updates)
            writer.add_scalar('SAC/policy', policy_loss, agent.policy_updates)

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory_novice.push(state, action, reward, next_state, mask)
        state = next_state

    writer.add_scalar('reward/episode_reward', episode_reward, i_episode)
    print("Episode: {}, reward: {}, final state: {}".format(i_episode, round(episode_reward, 2), state))

    if i_episode % 100 == 0:
        agent.save_model("gcl", str(i_episode))
        memory_novice.save_buffer("gcl", str(i_episode))
