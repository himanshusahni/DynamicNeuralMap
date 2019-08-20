import torch
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import numpy as np


class ReinforcePolicyDiscrete(object):
    """basic reinforce algorithm"""

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
        self.ep_rewards = []
        self.ep_log_probs = []
        self.gamma = 0.9
        self.eps = 1e-6
        self.attn_trans_matrix = np.array([[0, -10],
                                           [0,  10],
                                           [-10, 0],
                                           [10,  0]])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-6)

    def step(self, x, loc, test=False):
        """predict action based on input and update internal storage"""
        probs = self.policy(x)
        m = Categorical(probs)
        action = m.sample()
        if not test:
            self.ep_log_probs.append(m.log_prob(action))
        action = action.detach().cpu().numpy()
        action_one_hot = np.zeros((x.size(0), 4))
        action_one_hot[range(x.size(0)), action] = 1
        attn_trans = np.matmul(action_one_hot, self.attn_trans_matrix)
        loc += attn_trans.astype(np.int64)
        return loc

    def update(self):
        """called at the end of the episode to update policy"""
        R = 0
        policy_loss = []
        returns = []
        for r in self.ep_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + self.eps)
        for log_prob, R in zip(self.ep_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.ep_rewards = []
        self.ep_log_probs = []
        return policy_loss.item()


class ReinforcePolicyContinuous(object):
    """basic reinforce algorithm"""

    def __init__(self, input_size, attn_size, policy, device):
        self.input_size = float(input_size)
        self.attn_size = float(attn_size)
        self.policy = policy
        self.device = device
        self.ep_rewards = []
        self.ep_log_probs = []
        self.gamma = 0.9
        self.eps = 1e-6
        self.steps = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-6)

    def step(self, x, test=False):
        """predict action based on input and update internal storage"""
        probs = self.policy(x)
        m = MultivariateNormal(probs, 0.1*torch.eye(probs.size(1)).to(self.device))
        action = m.sample()
        one = torch.ones_like(action).to(self.device)
        action = torch.min(torch.max(action, -one), one)
        # with some probability, still take a random action
        randa = (torch.rand_like(action)*2 - 1).to(self.device)
        switch = torch.bernoulli(self.sigma * torch.ones_like(action)).to(self.device)
        action = switch * randa + (1 - switch) * action

        if not test:
            self.ep_log_probs.append(m.log_prob(action))
        # normalize to possible attention range
        # action *= ((self.input_size - self.attn_size + 1)/self.input_size)
        # normalize to image scale
        action = (action + 1)/2
        action = self.input_size * action.detach().cpu().numpy()
        return action

    @property
    def sigma(self):
        if self.steps < 200000:
            return 1 - self.steps*(1-0.05)/200000
        else:
            return 0.05

    def reward(self, r):
        self.ep_rewards.append(r)

    def update(self):
        """called at the end of the episode to update policy"""
        R = 0
        policy_loss = []
        returns = []
        for r in self.ep_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + self.eps)
        for log_prob, R in zip(self.ep_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.ep_rewards = []
        self.ep_log_probs = []
        self.steps += 1
        return policy_loss


class ReinforcePolicyRandom(object):
    """random glimpse locations"""

    def __init__(self, policy, device):
        pass

    def step(self, x, loc, test=False):
        action = np.random.rand(x.size(0), 2)
        # normalize to environment action range
        action = action * (44/64.) + (10/64.)
        action *= 64
        return action

    def update(self):
        return 0

    def reward(self, r):
        pass


