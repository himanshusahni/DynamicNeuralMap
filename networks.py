import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MapWrite_x_x(nn.Module):
    """what to write in static and dynamic parts of map"""

    def __init__(self, in_channels, out_channels):
        super(MapWrite_x_x, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing write network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)


class MapReconstruction_x_x(nn.Module):
    """reconstruct entire state from map"""

    def __init__(self, in_channels, out_channels):
        super(MapReconstruction_x_x, self).__init__()
        # decode the map back to original size
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing reconstruction network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        return torch.tanh(self.conv(x))


class MapWrite_84_21(nn.Module):
    """what to write in static and dynamic parts of map"""

    def __init__(self, in_channels, out_channels):
        super(MapWrite_84_21, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 8, stride=2, padding=3)  # 41x41
        self.res1 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.res2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)  # 21x21
        self.res3 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.res4 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing write network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp):
        inp = F.leaky_relu(self.conv1(inp), 0.2)
        # residual block
        x1 = F.leaky_relu(self.res1(inp), 0.2)
        x1 = self.res2(x1)
        inp = inp + x1
        inp = F.leaky_relu(self.conv2(inp), 0.2)
        # residual block
        x2 = F.leaky_relu(self.res3(inp), 0.2)
        x2 = self.res4(x2)
        inp = inp + x2
        return inp


class MapReconstruction_21_84(nn.Module):
    """reconstruct entire state from map"""

    def __init__(self, in_channels, out_channels):
        super(MapReconstruction_21_84, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1) # 21x21
        self.res1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.res2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)  # 42x42
        self.res3 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.res4 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, 8, stride=2, padding=3)  # 84x84
        self.print_info()

    def print_info(self):
        print("Initializing reconstruction network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp):
        inp = F.leaky_relu(self.conv1(inp), 0.2)
        # residual block
        x1 = F.leaky_relu(self.res1(inp), 0.2)
        x1 = self.res2(x1)
        inp = inp + x1
        inp = F.leaky_relu(self.conv2(inp), 0.2)
        # residual block
        x2 = F.leaky_relu(self.res3(inp), 0.2)
        x2 = self.res4(x2)
        inp = inp + x2
        inp = torch.tanh(self.conv3(inp))
        return inp


class MapBlend(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MapBlend, self).__init__()
        # convolve
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2)
        self.print_info()

    def print_info(self):
        print("Initializing blend network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp, map):
        x = torch.cat((inp, map), dim=1)
        return F.leaky_relu(self.conv1(x), 0.2)


class MapBlendSpatial(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MapBlendSpatial, self).__init__()
        # convolve
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.print_info()

    def print_info(self):
        print("Initializing blend network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp, map):
        x = torch.cat((inp, map), dim=1)
        return F.leaky_relu(self.conv1(x), 0.2)


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


class MapStepResidual(nn.Module):
    """Forward dynamics model for neural attention maps"""

    def __init__(self, in_channels, out_channels):
        super(MapStepResidual, self).__init__()
        # convolve
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(2 * in_channels, in_channels, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2)
        self.print_info()

    def print_info(self):
        print("Initializing step network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp):
        x = F.leaky_relu(self.conv1(inp), 0.2)
        x = torch.cat((x, inp), dim=1)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x


class MapStepResidualConditional(nn.Module):
    """Forward dynamics model for neural attention maps"""

    def __init__(self, in_channels, out_channels, nb_actions):
        super(MapStepResidualConditional, self).__init__()
        # convolve
        self.conv1 = torch.nn.Conv2d(in_channels + 4, in_channels, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(2 * in_channels, in_channels, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2)
        self.nb_actions = nb_actions
        self.print_info()

    def print_info(self):
        print("Initializing step network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, inp, a):
        a = a.unsqueeze(2).expand(-1, self.nb_actions, inp.size(2))
        a = a.unsqueeze(3).expand(-1, self.nb_actions, inp.size(2), inp.size(3))
        x = torch.cat((inp, a), dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = torch.cat((inp, x), dim=1)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x


class MapStepSpatial(nn.Module):
    """Forward dynamics model for spatial net memory"""

    def __init__(self, in_channels, out_channels):
        super(MapStepSpatial, self).__init__()
        # convolve
        self.conv1 = torch.nn.Conv2d(in_channels * 2, in_channels, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels * 2, in_channels, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2)
        self.print_info()

    def print_info(self):
        print("Initializing step network!")
        print(self)
        print("Total conv params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, map, inp):
        x = torch.cat((map, inp), dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = torch.cat((x, map), dim=1)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x


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


class ValueFunction(nn.Module):
    """value prediction layer on top of trunk above"""
    def __init__(self, channels, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv3 = nn.Conv2d(8, 4, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.fc1 = nn.Linear(input_size * input_size * 4, 1)
        self.print_info()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.flatten(start_dim=1)
        return self.fc1(x)

    def print_info(self):
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class ConvTrunk(nn.Module):

    def __init__(self, state_shape):
        super().__init__()
        channels = state_shape[0]
        input_size = state_shape[1]
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.output_size = input_size * input_size * 64

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x.flatten(start_dim=1)

    def print_info(self):
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class ConvTrunk3(nn.Module):

    def __init__(self, state_shape):
        super().__init__()
        channels = state_shape[0]
        input_size = state_shape[1]
        self.conv1 = nn.Conv2d(channels, 64, 4, stride=2, padding=2)
        input_size = np.ceil(input_size/2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        input_size = np.ceil(input_size/1)
        self.print_info()

        self.output_size = int(input_size * input_size * 64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return x.flatten(start_dim=1)

    def print_info(self):
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class ConvTrunk2(nn.Module):

    def __init__(self, state_shape):
        super().__init__()
        channels = state_shape[0]
        input_size = state_shape[1]
        self.conv1 = nn.Conv2d(channels, 32, 8, stride=4, padding=2)
        input_size = np.ceil(input_size/4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=2)
        input_size = np.ceil(input_size/2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        input_size = np.ceil(input_size/1)
        self.print_info()

        self.output_size = int(input_size * input_size * 64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x.flatten(start_dim=1)

    def print_info(self):
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class PolicyFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        channels = observation_space.shape[2]
        input_size = observation_space.shape[0]
        nb_actions = action_space.n
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.conv3 = nn.Conv2d(8, 4, 3, stride=2, padding=1)
        input_size = (input_size + 1) // 2
        self.fc1 = nn.Linear(input_size * input_size * 4, nb_actions)
        self.print_info()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.flatten(start_dim=1)
        return self.fc1(x)

    def print_info(self):
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class PolicyFunction_x_x(nn.Module):
    """policy prediction layer on top of trunk above"""
    def __init__(self, channels):
        super(PolicyFunction_x_x, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 1, 3, stride=1, padding=1)
        self.print_info()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return self.conv3(x).flatten(start_dim=1)

    def print_info(self):
        print("Initializing policy function of glimpse agent!")
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class PolicyFunction_21_84(nn.Module):
    """policy prediction layer on top of trunk above"""
    def __init__(self, channels):
        super(PolicyFunction_21_84, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, stride=1, padding=1) # 21x21
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear') # 42x42
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear') # 84x84
        self.conv3 = nn.Conv2d(8, 1, 3, stride=1, padding=1)
        self.print_info()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.up1(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.up2(x)
        return self.conv3(x).flatten(start_dim=1)

    def print_info(self):
        print("Initializing policy function of glimpse agent!")
        print(self)
        print("Total params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))


class ActorCritic_21_84(nn.Module):
    """actor critic container for separate 21_84 policy network and
    value network"""
    def __init__(self, trunk_arch, observation_space, action_space):
        super().__init__()
        self.policy_head = PolicyFunction_21_84(observation_space[0])
        self.value_head = ValueFunction(channels=observation_space[0], input_size=observation_space[1])

    def print_info(self):
        self.policy_head.print_info()
        self.value_head.print_info()

    def parameters(self, recurse=True):
        return list(self.policy_head.parameters()) + \
               list(self.value_head.parameters())

    def V_parameters(self, recurse=True):
        return list(self.value_head.parameters())

    def pi_parameters(self, recurse=True):
        return list(self.policy_head.parameters())

    def forward(self, x):
        return self.policy_head(x), self.value_head(x)

    def V(self, x):
        """only return values"""
        return self.value_head(x)

    def pi(self, x):
        """only return policy logits"""
        return self.policy_head(x)

    def to(self, device):
        self.policy_head.to(device)
        self.value_head.to(device)


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
        # return self.fc3(x), self.fc4(x)
        return self.fc3(x)

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
