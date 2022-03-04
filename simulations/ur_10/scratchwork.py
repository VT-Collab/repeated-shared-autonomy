import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from sklearn.utils import shuffle
import sys
import os
import copy 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Clear GPU memory from previous runs
if device == "cuda":
    torch.cuda.empty_cache()

data1 = pickle.load(open("demos/push1_1.pkl","rb"))
data2 = pickle.load(open("demos/push1_2.pkl","rb"))
latent_dim = 5
#RNN
gru = nn.GRU(
    input_size = 18,
    hidden_size = latent_dim,
    num_layers = 1,
    batch_first = True
    )
fcn = nn.Sequential(
    nn.Linear(latent_dim, 5),
    nn.Tanh(),
    nn.Linear(5, 2)
)
traj = [item[0] + item[1] for item in data1]

i1 = torch.Tensor(traj).unsqueeze(0)
# i1 = i1.unsqueeze(0)
target = np.zeros(len(traj))
print(target.shape)
# traj = [item[0] + item[1] for item in data2]
# i2 = torch.Tensor(traj)
# i2 = i2.unsqueeze(0)

# input = torch.cat((i1, i2), dim=0)
# print(input.shape)

# h, o = gru(input)
# y = fcn(h)
# print(y)