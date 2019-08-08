import torch
import torch.nn as nn
import torch.nn.functional as F


class MapReconstruction(nn.Module):
    """reconstruct entire state from map"""

    def __init__(self, in_channels, out_channels):
        super(MapReconstruction, self).__init__()
        # decode the map back to original size
        # self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=1, padding=1)
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

    def __init__(self, channels):
        super(MapStep, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing step network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        return torch.tanh(self.conv(x))


class MapWrite(nn.Module):
    """what to write in static and dynamic parts of map"""

    def __init__(self, attn_size, in_channels, out_channels):
        super(MapWrite, self).__init__()
        self.attn_size = attn_size
        self.fc1 = nn.Linear(attn_size*attn_size*in_channels, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.print_info()

    def print_info(self):
        print("Initializing write network!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        w = F.leaky_relu(self.fc2(x), 0.2)
        return w


class GlimpseNetwork(nn.Module):
    """using embedding from above decides where the next glimpse should be"""

    def __init__(self, nb_actions):
        super(GlimpseNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, nb_actions)
        self.print_info()

    def print_info(self):
        print("Initializing glimpse network!")
        print(self)
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        """predict action"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        return torch.tanh(self.fc2(x))


class Trunk(nn.Module):
    """fully connected common trunk for value and policy functions"""

    def __init__(self, obs_size):
        super(Trunk, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        return F.leaky_relu(self.fc2(x), 0.2)


class ValueFunction(nn.Module):
    """value prediction layer on top of trunk above"""
    def __init__(self, trunk):
        super(ValueFunction, self).__init__()
        self.trunk = trunk
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc1(self.trunk(x))


class PolicyFunction(nn.Module):
    """policy prediction layer on top of trunk above"""
    def __init__(self, trunk, nb_actions):
        super(PolicyFunction, self).__init__()
        self.trunk = trunk
        self.fc1 = nn.Linear(64, nb_actions)

    def forward(self, x):
        return self.fc1(self.trunk(x))
