import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MapReconstruction(nn.Module):
    """reconstruct entire state from map"""

    def __init__(self, in_channels, out_channels):
        super(MapReconstruction, self).__init__()
        # decode the map back to original size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing reconstruction network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        return torch.tanh(self.conv(x))


class MapStep(nn.Module):
    """Forward dynamics model for neural attention maps"""

    def __init__(self, in_channels, out_channels):
        super(MapStep, self).__init__()
        # convolve
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels+out_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing step network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp):
        x = F.leaky_relu(self.conv1(inp), 0.2)
        # residual connection
        x = torch.cat((inp, x), dim=1)
        return F.leaky_relu(self.conv2(x), 0.2)


class MapStepRot(nn.Module):
    """Forward dynamics model for neural attention maps"""

    def __init__(self, channels):
        super(MapStepRot, self).__init__()
        # convolve
        self.conv_weights = torch.nn.Parameter(data=torch.Tensor(channels, channels, 9), requires_grad=True)
        conv_init = np.random.rand(channels, 1, 9)
        self.conv_weights.data = torch.from_numpy(conv_init.astype(np.float32))
        self.channels = channels
        # rotate
        self.rot_weights = torch.nn.Parameter(data=torch.Tensor(channels, channels, 1, 1), requires_grad=True)
        orthonormal_init = np.linalg.qr(np.random.rand(channels, channels))[0]
        self.rot_weights.data = torch.from_numpy(orthonormal_init.astype(np.float32)).view(channels, channels, 1, 1)
        self.print_info()

    def print_info(self):
        print("Initializing step network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        weights = F.softmax(self.conv_weights, dim=-1).view(self.channels, 1, 3, 3)
        x = F.conv2d(x, weights, padding=1, groups=self.channels)
        # rot now
        return F.conv2d(x, self.rot_weights)


class MapWrite(nn.Module):
    """what to write in static and dynamic parts of map"""

    def __init__(self, attn_size, in_channels, out_channels):
        super(MapWrite, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing write network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)


class Trunk(nn.Module):
    """fully connected common trunk for value and policy functions"""

    def __init__(self, input_size, channels):
        super(Trunk, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.output_size = (32, input_size // 4 + 1, input_size // 4 + 1)

    def forward(self, x):
        """predict action"""
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return x.flatten(start_dim=1)

    def print_info(self):
        print("Initializing trunk of glimpse network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class FCTrunk(nn.Module):
    """fully connected common trunk for value and policy functions"""

    def __init__(self, input_size):
        super(FCTrunk, self).__init__()
        self.fc1= nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output_size = (32,)

    def forward(self, x):
        """predict action"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return x

    def print_info(self):
        print("Initializing trunk of glimpse network!")
        print(self)
        print("Total fc params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class ValueFunction(nn.Module):
    """value prediction layer on top of trunk above"""
    def __init__(self, channels, input_size):
        super(ValueFunction, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.fc1 = nn.Linear(input_size * input_size * 16, 32)
        self.fc2 = nn.Linear(32, 1)
        self.print_info()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.flatten(start_dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        return self.fc2(x)

    def print_info(self):
        print("Initializing value function of glimpse agent!")
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

class FCValueFunction(nn.Module):
    """value prediction layer on top of trunk above"""
    def __init__(self, input_size):
        super(FCValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class PolicyFunction(nn.Module):
    """policy prediction layer on top of trunk above"""
    def __init__(self, channels):
        super(PolicyFunction, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.print_info()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return self.conv3(x).flatten(start_dim=1)

    def print_info(self):
        print("Initializing policy function of glimpse agent!")
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class FCPolicyFunction(nn.Module):
    """policy prediction layer on top of trunk above"""
    def __init__(self, input_size, nb_actions):
        super(FCPolicyFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, nb_actions)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x), self.fc4(x)
        # return self.fc3(x)
