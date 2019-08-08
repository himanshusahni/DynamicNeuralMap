import torch
import torch.optim as optim

import numpy as np

from networks import *


class DynamicMap():
    def __init__(self, size, channels, attn_size, device):
        self.size = size
        self.attn_size = attn_size
        self.write_model = MapWrite(attn_size, in_channels=channels, out_channels=16)
        self.step_model = MapStep(channels=8)
        self.reconstruction_model = MapReconstruction(in_channels=16, out_channels=channels)
        self.device = device

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
        # net.hidden = (torch.zeros(1, BATCH_SIZE, 64).to(device),
        #               torch.zeros(1, BATCH_SIZE, 64).to(device))

    def write(self, glimpse, attn):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        w = self.write_model(glimpse)
        # write
        batchsize = glimpse.size(0)
        attn_size = glimpse.size(2)
        idxs_dim_0 = np.repeat(np.arange(batchsize), attn_size*attn_size)
        idxs_dim_2 = attn[:, 0, :].flatten()
        idxs_dim_3 = attn[:, 1, :].flatten()
        # more indices magic
        w = w.transpose(0, 1).flatten(start_dim=1).transpose(0, 1)
        self.map[idxs_dim_0, idxs_dim_2, idxs_dim_3, :] = w
        # returns a cost of writing
        return w.abs().mean()

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
