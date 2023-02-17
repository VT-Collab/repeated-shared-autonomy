import argparse
import datetime
from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from learn_intent import RobotInterface

parser = argparse.ArgumentParser()
parser.add_argument('--play', default="None")
parser.add_argument('--num_eps', type=int, default=100)
parser.add_argument('--start_eps', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--memory_size', type=int, default=10000)
args = parser.parse_args()


# Environment
env = RobotInterface()

# Agent
agent = SAC(env.action_space)

# Tensorboard
folder = "runs/expert/"
writer = SummaryWriter(folder + '{}'.format(datetime.datetime.now().strftime("%m-%d_%H-%M")))

# Memory
memory_expert = ReplayMemory(capacity=args.memory_size)

# Play expert policy
if args.play != "None":
    agent.load_model("expert", args.play)
    args.start_eps = 0

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

        if len(memory_expert) > args.batch_size and args.play == "None":
            critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory_expert, args.batch_size)
            writer.add_scalar('SAC/critic_1', critic_1_loss, agent.policy_updates)
            writer.add_scalar('SAC/critic_2', critic_2_loss, agent.policy_updates)
            writer.add_scalar('SAC/policy', policy_loss, agent.policy_updates)

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory_expert.push(state, action, reward, next_state, mask)
        state = next_state

    writer.add_scalar('reward/episode_reward', episode_reward, i_episode)
    print("Episode: {}, reward: {}, final state: {}".format(i_episode, round(episode_reward, 2), state))

    if i_episode % 100 == 0 and args.play == "None":
        agent.save_model("expert", str(i_episode))
    if i_episode % 10 == 0 and args.play != "None":
        memory_expert.save_buffer("expert", str(i_episode))
