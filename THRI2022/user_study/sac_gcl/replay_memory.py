import random
import numpy as np
import pickle
import copy
import os


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # push one timestep of data
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # sample a batch of datapoints
    def sample(self, batch_size, flag=False):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    # Get number of interactions
    def __len__(self):
        return len(self.buffer)

    # Save the memory buffer
    def save_buffer(self, algo, name):
        print('[*] Saving as models/{}/buffer_{}_{}.pkl'.format(algo, algo, name))
        if not os.path.exists('models/{}/'.format(algo)):
            os.makedirs('models/{}/'.format(algo))
        with open("models/{}/buffer_{}_{}.pkl".format(algo, algo, name), 'wb') as f:
            pickle.dump(self.buffer, f)

    # Load the memory buffer
    def load_buffer(self, algo, name):
        print('[*] Loading from models/{}/buffer_{}_{}.pkl'.format(algo, algo, name))
        with open("models/{}/buffer_{}_{}.pkl".format(algo, algo, name), "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
