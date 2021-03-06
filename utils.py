import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import torch
from torch.utils.data import Dataset


class CurriculumDataset(Dataset):
    """loading expert trajectory data"""
    def __init__(self, demo_dir, preload=False):
        """
        :param demo_dir: directory containing observation episodes 0/, 1/...
        :param start_len: length of sequence at start of training
        :param end_len: maximum length of a sequence
        """
        self.demo_dir = demo_dir
        self.preload = preload
        # iterate through all episodes and count up steps in each
        self.nb_eps = len(os.listdir(demo_dir))

        self.ep_mapping = []
        self.ep_steps = [0]
        if preload:
            self.imgs = []
            self.actions = []
        # for ep in range(150):
        for ep in range(self.nb_eps):
            ep_dir = os.path.join(demo_dir, str(ep))
            # nb samples is episode length - 1 bcs. predicting next state
            nb_samples = len([f for f in os.listdir(ep_dir)
                              if f.endswith('.pt') and ('reward' not in f) and 'action' not in f]) - 1
            self.ep_mapping += [ep]*nb_samples
            self.ep_steps.append(self.ep_steps[-1] + nb_samples)
            if preload:
                print("preloading ep {}/{}".format(ep, self.nb_eps))
                self.imgs.append([torch.load(os.path.join(ep_dir, '{}.pt'.format(step))) for step in range(nb_samples+1)])
                self.actions.append(torch.load(os.path.join(ep_dir, 'actions.pt')))

    def __len__(self):
        return self.nb_eps

    def set_seqlen(self, seq_len):
        self.seq_len = seq_len

    def __getitem__(self, ep):
        """get a random snippet from this episode"""
        end_idx = None
        while end_idx is None:
            try:
                end_idx = np.random.randint(self.seq_len, self.ep_steps[ep+1] - self.ep_steps[ep])
            except ValueError:
                ep = np.random.randint(self.nb_eps)
        start_idx = max(0, end_idx-self.seq_len)
        if self.preload:
            imgs = self.imgs[ep][start_idx:end_idx]
            actions = self.actions[ep][start_idx:end_idx]
        else:
            # load the imgs from disk
            imgs = []
            for idx in range(start_idx, end_idx):
                img_name = os.path.join(
                    self.demo_dir,
                    str(ep),
                    '{}.pt'.format(idx))
                imgs.append(torch.load(img_name))
            actions = torch.load(os.path.join(self.demo_dir, str(ep),'actions.pt'))
            actions = actions[start_idx:end_idx]
        imgs = torch.cat([img.unsqueeze(dim=0) for img in imgs])
        return {'imgs': imgs, 'actions': actions}


def time_collate(batch):
    imgs = torch.cat([sample['imgs'].unsqueeze(dim=0) for sample in batch])
    imgs = imgs.transpose(0, 1)
    actions = torch.cat([sample['actions'].unsqueeze(dim=0) for sample in batch])
    actions = actions.transpose(0, 1)
    return imgs, actions


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, reset_freq=1000):
        self.reset_freq = reset_freq
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count % self.reset_freq == 0:
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess(img):
    """
    :param img: batch of pytorch images in [-1,1]
    :return: numpy friendly image in [0,255]
    """
    return (((np.transpose(img, (0, 2, 3, 1))+1)/2.)*255).astype(np.uint8)


def save_example_images(test_batch, test_maps_prestep, test_maps_poststep, test_locs, path, env):
    """creates a display friendly visualization of results"""
    attn_size = 3
    size = 10
    fig, axarr = plt.subplots(len(test_maps_prestep), 3, figsize=(2*3, 2*len(test_maps_prestep)))
    for t in range(len(test_maps_prestep)):
        display_state = env.render(test_batch[t].squeeze().permute(1, 2, 0).numpy())
        axarr[t, 0].imshow(display_state)
        axarr[t, 0].set_xticks([])
        axarr[t, 0].set_yticks([])
        # now make the attention visible
        attention = patches.Rectangle(
            ((test_locs[t,0]-attn_size//2)*40, (size-1-test_locs[t,1]-attn_size//2)*40),
            attn_size*40, attn_size*40, linewidth=2, edgecolor='y', facecolor='none')
        axarr[t, 0].add_patch(attention)
        display_prestep = env.render(test_maps_prestep[t].permute(1, 2, 0).numpy())
        axarr[t, 1].imshow(display_prestep)
        axarr[t, 1].set_xticks([])
        axarr[t, 1].set_yticks([])
        display_poststep = env.render(test_maps_poststep[t].permute(1, 2, 0).numpy())
        axarr[t, 2].imshow(display_poststep)
        axarr[t, 2].set_xticks([])
        axarr[t, 2].set_yticks([])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(path, bbox_inches=0)
