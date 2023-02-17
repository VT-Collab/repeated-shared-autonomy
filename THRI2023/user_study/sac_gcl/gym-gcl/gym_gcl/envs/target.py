import numpy as np
import gym
from gym import spaces



class Target(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(
            low=-0.2,
            high=+0.2,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(6,),
            dtype=np.float32
        )


    def _get_obs(self):
        return np.array([self.robot[0], self.robot[1], self.robot[2], self.target[0], self.target[1], self.target[2]])


    def reset(self):
        self.robot = np.array([0., 0., 0.])
        self.target = np.random.uniform(-2., 2., 3)
        return self._get_obs()


    def step(self, action):
        self.robot += action
        reward = -np.linalg.norm(self.robot - self.target) * 100
        done = False
        return self._get_obs(), reward, done, {}
