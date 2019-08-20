import torch
import torch.optim as optim

import numpy as np

from networks import *
from utils import *


class DynamicMap():
    def __init__(self, size, channels, attn_size, device):
        self.size = size
        self.attn_size = attn_size
        self.write_model = MapWrite(attn_size, in_channels=channels, out_channels=16)
        self.step_model = MapStep(channels=8)
        self.reconstruction_model = MapReconstruction(in_channels=16, out_channels=channels)
        self.device = device
        self.mse = MSEMasked()

    def to(self, device):
        self.write_model.to(device)
        self.step_model.to(device)
        self.reconstruction_model.to(device)

    def reset(self, batchsize):
        """
        reset the map to beginning of episode
        :param batchsize: batch size of observations that will be used before next reset
        :return:
        """
        self.map = torch.zeros((batchsize, self.size, self.size, 16)).to(self.device)

    def prepare_write(self, glimpse, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        w = self.write_model(glimpse).permute(0, 2, 3, 1)
        # write
        map_mask = obs_mask.permute(0, 2, 3, 1)
        w *= map_mask
        w_mse = self.mse(w.detach(), self.map, map_mask)
        return w, w.abs().sum()/map_mask.sum(), w_mse

    def write(self, w, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # write
        map_mask = obs_mask.permute(0, 2, 3, 1)
        minus_map_mask = minus_obs_mask.permute(0, 2, 3, 1)
        self.map = self.map * minus_map_mask + w

    def step(self, action):
        """
        uses the model to advance the map by a step
        :param action: (batchsize,) actions taken by agent
        """
        # only dynamic part of map is stepped
        dynamic = self.map[:, :, :, 8:]
        dynamic = self.step_model(dynamic.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        self.map[:, :, :, 8:] = dynamic

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
        models = torch.load(path)
        self.write_model = models['write']
        self.step_model = models['step']
        self.reconstruction_model = models['reconstruct']
