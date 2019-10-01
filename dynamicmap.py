import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from networks import *


class DynamicMap():
    def __init__(self, size, channels, env_size, env_channels, batchsize, device):
        self.size = size
        self.channels = channels
        self.env_size = env_size
        self.batchsize = batchsize
        self.device = device
        if env_size == 10 and size == 10:
            self.write_model = MapWrite_x_x(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_x_x(in_channels=channels, out_channels=env_channels)
        elif env_size == 84 and size == 21:
            self.write_model = MapWrite_84_21(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_21_84(in_channels=channels, out_channels=env_channels)
        elif env_size == 84 and size == 84:
            self.write_model = MapWrite_84_84(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_84_84(in_channels=channels, out_channels=env_channels)
        self.step_model = MapStepResidual(in_channels=channels, out_channels=channels//2)

    def to(self, device):
        self.write_model.to(device)
        self.step_model.to(device)
        self.reconstruction_model.to(device)

    def reset(self):
        """
        reset the map to beginning of episode
        """
        self.map = torch.zeros((self.batchsize, self.size, self.size, self.channels)).to(self.device)

    def MaskObs2Map(self, mask, minus_mask):
        '''
        converts an observation mask to one of size of map
        '''
        if self.size == self.env_size:
            minus_map_mask = minus_mask
            map_mask = mask
        elif self.size == self.env_size // 2:
            minus_map_mask = nn.MaxPool2d(2, stride=2)(minus_mask)
            map_mask = 1 - minus_map_mask
        elif self.size == self.env_size // 4:
            minus_map_mask = nn.MaxPool2d(2, stride=2)(minus_mask)
            minus_map_mask = nn.MaxPool2d(2, stride=2)(minus_map_mask)
            map_mask = 1 - minus_map_mask
        map_mask = map_mask.permute(0, 2, 3, 1).detach()
        minus_map_mask = minus_map_mask.permute(0, 2, 3, 1).detach()
        return map_mask, minus_map_mask

    def write(self, glimpse, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        w = self.write_model(glimpse).permute(0, 2, 3, 1)
        # write
        map_mask, minus_map_mask = self.MaskObs2Map(obs_mask, minus_obs_mask)
        w *= map_mask
        self.map = self.map * minus_map_mask + w
        # returns a cost of writing
        return w.abs().mean()

    def step(self):
        """
        uses the model to advance the map by a step
        """
        # only dynamic part of map is stepped, the whole map is provided as input
        dynamic = self.step_model(self.map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        self.map[:, :, :, self.channels//2:] = self.map[:, :, :, self.channels//2:] + dynamic
        return dynamic.abs().mean()

    def reconstruct(self):
        """
        attempt to reconstruct the entire state image using current map
        """
        return self.reconstruction_model(self.map.permute(0, 3, 1, 2))

    def parameters(self):
        return list(self.write_model.parameters()) +\
               list(self.step_model.parameters()) +\
               list(self.reconstruction_model.parameters())

    def save(self, path):
        torch.save({
            'write': self.write_model,
            'step': self.step_model,
            'reconstruct': self.reconstruction_model
        }, path)

    def load(self, path):
        models = torch.load(path, map_location='cpu')
        self.write_model = models['write']
        self.step_model = models['step']
        self.reconstruction_model = models['reconstruct']
