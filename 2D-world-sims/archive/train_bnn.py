import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

# collect dataset
class MotionData(Dataset):

    def __init__(self, filename):
        self.all_data = pickle.load(open(filename, "rb"))
        self.max_len = len(self.all_data)
        self.train_len = int(0.75 * self.max_len)
        self.data = random.sample(self.all_data, self.train_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        state = torch.FloatTensor(item[1]).to(device)
        true_z = torch.FloatTensor(item[2]).to(device)
        action = torch.FloatTensor(item[3]).to(device)
        return (snippet, state, true_z, action)

class Gaussian(object):
    def __init__(self,mu,rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)



class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = BayesianLinear(10, 10)
        self.e2 = BayesianLinear(10, 12)
        self.e3 = BayesianLinear(12, 10)

        self.fc_mean = BayesianLinear(10, 1)
        self.fc_var = BayesianLinear(10, 1)
    
        self.d1 = BayesianLinear(9, 12)
        self.d2 = BayesianLinear(12, 10)
        self.d3 = BayesianLinear(10, 2)

    def forward(self, x):

        return x

    def encoder(self, x):
        x = F.relu(self.e1(x, sample))
        x = F.relu(self.e2(x, sample))
        x = F.log_softmax(self.e3(x, sample), dim=1)
    
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    # def sample_elbo(self, input, target, samples=SAMPLES):
    #     outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(device)
    #     log_priors = torch.zeros(samples).to(device)
    #     log_variational_posteriors = torch.zeros(samples).to(device)
    #     for i in range(samples):
    #         outputs[i] = self(input, sample=True)
    #         log_priors[i] = self.log_prior()
    #         log_variational_posteriors[i] = self.log_variational_posterior()
    #     log_prior = log_priors.mean()
    #     log_variational_posterior = log_variational_posteriors.mean()
    #     negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
    #     loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood
    #     return loss, log_prior, log_variational_posterior, negative_log_likelihood

# net = BayesianNetwork().to(device)

# conditional autoencoder
class BVAE(nn.Module):

    def __init__(self):
        super(BVAE, self).__init__()

        self.loss_func = nn.MSELoss()
        self.BETA = 0.001

        # encoder
        self.e1 = BayesianLinear(10, 10)
        self.e2 = BayesianLinear(10, 12)
        self.e3 = BayesianLinear(12, 10)

        self.fc_mean = BayesianLinear(10, 1)
        self.fc_var = BayesianLinear(10, 1)
    
        # decoder
        self.d1 = BayesianLinear(9, 12)
        self.d2 = BayesianLinear(12, 10)
        self.d3 = BayesianLinear(10, 2)

    def reparam(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encoder(self, x):
        x = F.relu(self.e1(x, sample))
        x = F.relu(self.e2(x, sample))
        h = F.log_softmax(self.e3(x, sample), dim=1)
        return self.fc_mean(h), self.fc_var(h)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        c = x[0]
        s = x[1]
        ztrue = x[2] #this is the ground truth label, for use when trouble shooting
        a = x[3]
        mean, log_var = self.encoder(c)
        z = self.reparam(mean, log_var)
        z_with_state = torch.cat((z, s), 1)
        a_decoded = self.decoder(z_with_state)
        loss = self.loss(a, a_decoded, mean, log_var)
        return loss

    def loss(self, a_decoded, a_target, mean, log_var):
        rce = self.loss_func(a_decoded, a_target)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return rce + self.BETA * kld

def main(num):

    model = BVAE()
    model = model.to(device)
    dataname = 'data/dataset.pkl'
    savename = "models/bvae_" + str(num)
    print(savename)

    EPOCH = 2000
    BATCH_SIZE_TRAIN = 400
    LR = 0.01
    LR_STEP_SIZE = 1400
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
    torch.save(model.state_dict(), savename)

if __name__ == "__main__":
    # for i in range(1,11):
    main(1)