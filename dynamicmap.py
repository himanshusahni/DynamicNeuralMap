import torch
import torch.optim as optim

import numpy as np

from networks import *


class DynamicMap():
    def __init__(self, size, attn_size, device):
        self.size = size
        self.attn_size = attn_size
        self.write_model = MapWrite(attn_size, in_channels=5, out_channels=16)
        self.step_model = MapStep(channels=8)
        self.reconstruction_model = MapReconstruction(in_channels=16, out_channels=5)
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

    def register_glimpse(self, glimpse, attn):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attm: (x,y) location of above glimpse in map coordinates
        """
        # what to write, w_s (batchsize, 8) w_d (batchsize, 8) and with what probability (batchsize, attn_size^2)
        w, p = self.write_model(glimpse)
        # p = torch.ones(glimpse.size(0), self.attn_size * self.attn_size)
        # p = p.to(self.device)
        w = w.repeat(1, self.attn_size*self.attn_size).view(-1, self.attn_size, self.attn_size, 16)
        w *= p.view(-1, self.attn_size, self.attn_size, 1)
        # write into map
        for idx in range(attn.shape[0]):
            self.map[idx, attn[idx, 0, :], attn[idx, 1, :]] = w[idx].view(-1, 16)

    def step(self, action):
        """
        uses the model to advance the map by a step
        :param action: (batchsize,) actions taken by agent
        """
        # only dyanamic part of map is stepped
        dynamic = self.map[:, :, :, 8:]
        dynamic = self.step_model(dynamic.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        self.map[:, :, :, 8:] = dynamic

    def reconstruct(self):
        """
        attempt to reconstruct the entire state image using current map
        """
        return self.reconstruction_model(self.map).permute(0, 3, 1, 2)

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

    def load(self, path, device):
        models = torch.load(path)
        self.write_model = models['write'].to(device)
        self.step_model = models['step'].to(device)
        self.reconstruction_model = models['reconstruct'].to(device)
