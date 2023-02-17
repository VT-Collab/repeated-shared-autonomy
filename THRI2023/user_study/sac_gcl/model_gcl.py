
import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class CostNN(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(CostNN, self).__init__()

        self.ReLU = nn.LeakyReLU()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim/2)
        self.linear3 = nn.Linear(hidden_dim/2, hidden_dim/4)
        self.linear4 = nn.Linear(hidden_dim/4, 1)

        self.apply(weights_init_)

    def forward(self, state):

        h1 = self.ReLU(self.linear1(state))
        h2 = self.ReLU(self.linear2(h1))
        h2 = self.ReLU(self.linear3(h2))
        return torch.tanh(self.linear4(h2))
