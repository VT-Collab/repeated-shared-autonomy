import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from experts.PG import PG
import time

from torch.optim.lr_scheduler import StepLR
# seed = 18095048
# seed = int(time.time())
seed = 1666219353
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode=None).unwrapped
# env = ContinuousCartPoleEnv()
if seed is not None:
    env.action_space.seed(seed)
state = env.reset()

n_actions = env.action_space.shape[0]
state_shape = env.observation_space.shape

# initializing a model
model = PG(state_shape, n_actions)

mean_rewards = []
for i in range(100):
    rewards = [model.train_on_env(env) for _ in range(100)] 
    mean_rewards.append(np.mean(rewards))
    if i % 5:
        print("mean reward:%.3f" % (np.mean(rewards)))
        plt.figure(figsize=[9, 6])
        plt.title("Mean reward per 100 games")
        plt.plot(mean_rewards)
        plt.grid()
        # plt.show()
        plt.savefig('plots/PG_learning_curve.png')
        plt.close()
    
    # if np.mean(rewards) > 500:
    #     print("TRAINED!")
    #     break
    if np.mean(rewards) > -3000:
        print("TRAINED!")
        break

torch.save(model, "experts/saved_expert/pg_pendulum.model")
#model.load("experts/saved_expert/pg.model")

num_expert = 1000

expert_samples = np.array([model.generate_session(env) for _ in range(num_expert)])
np.save('expert_samples/pg_pendulum', expert_samples)
