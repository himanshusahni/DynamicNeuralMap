import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks import *
from utils import *
from reinforce import *
from dynamicmap import DynamicMap

import time
import numpy as np
import os
np.set_printoptions(precision=3)

env_name = 'DynamicObjects-v0'
from dynamicobject import DynamicObjects
env = DynamicObjects(size=10)

# args:
SEED = 354
ATTN_SIZE = 3
ENV_SIZE = env.observation_space.shape[0]
CHANNELS = env.observation_space.shape[2]
BATCH_SIZE = 4

np.random.seed(SEED)
torch.manual_seed(SEED)

# initialize training data
demo_dir = 'data-{}/'.format(env_name)
print('using training data from {}'.format(demo_dir))
dataset = CurriculumDataset(demo_dir=demo_dir, preload=False)
seq_len = 64
dataset.set_seqlen(seq_len)
test_idxs = list(range(len(dataset)))
test_idx_sampler = SubsetRandomSampler(test_idxs)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=test_idx_sampler, num_workers=1, collate_fn=time_collate,
                         drop_last=True)
test_loader_iter = iter(test_loader)

# gpu?
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("will run on {} device!".format(device))

# initialize map
map = DynamicMap(size=ENV_SIZE, channels=CHANNELS, attn_size=ATTN_SIZE, device=device)
map.to(device)
mse = MSEMasked()
mse_unmasked = nn.MSELoss(reduction='none')
# saved models
name = 'fullyconvglimpseagent_mapinput_newstep3'
model_dir = '{}/{}/'.format(env_name, name)
model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir)
               if name.endswith('pth') and
               # ('100000' in name or '150000' in name or '200000' in name or '250000' in name or '300000' in name) and
               # '630000' in name and
               'map' in name]

attn_span = range(-(ATTN_SIZE//2), ATTN_SIZE//2+1)
xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)

start = time.time()
idxs_dim_0 = np.repeat(np.arange(BATCH_SIZE), ATTN_SIZE * ATTN_SIZE)
for path in model_paths:
    # load the model
    map.load(path)
    it = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1][3:])
    # load the glimpse agent
    pathdir, pathname = os.path.split(path)
    glimpsenet = torch.load(os.path.join(pathdir, pathname.replace("map", "glimpsenet")), map_location='cpu')
    glimpse_agent = A2CPolicy(ENV_SIZE, glimpsenet['policy_network'], glimpsenet['value_network'], device)
    map.to(device)
    # draw a testing batch
    try:
        test_batch = next(test_loader_iter)
    except StopIteration:
        test_loader_iter = iter(test_loader)
        test_batch = next(test_loader_iter)
    state_batch, action_batch = test_batch
    # send to gpu
    state_batch = state_batch.to(device)
    attn_log_probs = []
    attn_rewards = []
    test_maps_prestep = []
    test_maps_heatmaps = []
    test_maps_poststep = []
    test_locs = []
    write_loss = 0
    post_write_loss = []
    post_step_loss = []
    sigma = []
    overall_reconstruction_loss = []
    # start!
    map.reset(batchsize=BATCH_SIZE)
    # get an empty reconstruction
    post_step_reconstruction = map.reconstruct()
    loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2), random=False)
    logits = glimpse_agent.pi(map.map.detach().permute(0, 3, 1, 2).to(device))
    test_maps_heatmaps.append(F.softmax(logits[0], dim=-1).view(1, ENV_SIZE, ENV_SIZE).detach().cpu())
    loc = np.clip(loc, 1, 8).astype(np.int64)  # clip to avoid edges
    test_locs.append(loc[0].copy())
    attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
    idxs_dim_2 = attn[:, 0, :].flatten()
    idxs_dim_3 = attn[:, 1, :].flatten()
    obs_mask = torch.zeros(BATCH_SIZE, 1, ENV_SIZE, ENV_SIZE)
    obs_mask[idxs_dim_0, :, idxs_dim_2, idxs_dim_3] = 1
    obs_mask = obs_mask.to(device)
    minus_obs_mask = 1-obs_mask
    for t in range(1, seq_len):
        post_step_reconstruction = post_step_reconstruction * minus_obs_mask + state_batch[t-1] * obs_mask
        write_loss += map.write(post_step_reconstruction.detach(), obs_mask, minus_obs_mask).item()
        post_write_reconstruction = map.reconstruct()
        post_write_loss.append(mse(post_write_reconstruction, state_batch[t-1], obs_mask).detach().cpu().numpy())
        test_maps_prestep.append(post_write_reconstruction[0].detach().cpu())
        # step forward the internal map
        map.step(action_batch[t-1])
        post_step_reconstruction = map.reconstruct()
        # select next attention spot
        loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2), random=False)
        logits = glimpse_agent.pi(map.map.detach().permute(0, 3, 1, 2).to(device))
        test_maps_heatmaps.append(F.softmax(logits[0], dim=-1).view(1, ENV_SIZE, ENV_SIZE).detach().cpu())
        sigma.append(glimpse_agent.policy.entropy(logits).detach().cpu().numpy())
        loc = np.clip(loc, 1, 8).astype(np.int64)  # clip to avoid edges
        test_locs.append(loc[0].copy())
        attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
        idxs_dim_0 = np.repeat(np.arange(BATCH_SIZE), ATTN_SIZE * ATTN_SIZE)
        idxs_dim_2 = attn[:, 0, :].flatten()
        idxs_dim_3 = attn[:, 1, :].flatten()
        next_obs_mask = torch.zeros(BATCH_SIZE, 1, ENV_SIZE, ENV_SIZE)
        next_obs_mask[idxs_dim_0, :, idxs_dim_2, idxs_dim_3] = 1
        next_obs_mask = next_obs_mask.to(device)
        # compute reconstruction loss
        post_step_loss.append(mse(post_step_reconstruction, state_batch[t], next_obs_mask).detach().cpu().numpy())
        # calculate per-channel losses on overall image
        channel_loss = []
        for ch in range(CHANNELS):
            l = mse_unmasked(post_step_reconstruction[:, ch].flatten(start_dim=1), state_batch[t][:, ch].flatten(start_dim=1))
            l = l.mean(dim=1)
            channel_loss.append(l.detach().cpu().numpy())
        overall_reconstruction_loss.append(channel_loss)
        test_maps_poststep.append(post_step_reconstruction[0].detach().cpu())
        obs_mask = next_obs_mask
        minus_obs_mask = 1-obs_mask
    overall_reconstruction_loss = torch.FloatTensor(overall_reconstruction_loss)
    post_step_loss = torch.FloatTensor(post_step_loss)
    post_write_loss = torch.FloatTensor(post_write_loss)
    sigma = torch.FloatTensor(sigma)
    test_loss = 0.01 * write_loss + post_write_loss.mean() + post_step_loss.mean()
    model_name = os.path.splitext(os.path.basename(path))[0]
    torch.save({
        'write_loss': 0.01 * write_loss,
        'post write reconstruction loss': post_write_loss,
        'post step reconstruction loss': post_step_loss,
        'sigma': sigma,
        'overall_reconstruction_loss': overall_reconstruction_loss,
    }, os.path.join(env_name, name, 'loss_{}.pt'.format(model_name)))
    # save some generated images
    save_example_images(
        [state_batch[t][0].cpu() for t in range(seq_len)],
        test_maps_heatmaps,
        test_maps_prestep,
        test_maps_poststep,
        test_locs,
        os.path.join(env_name, name, 'predictions_{}.jpeg'.format(model_name)),
        env)
    to_print = "[{}] test loss: {:.3f}".format(model_name, test_loss)
    to_print += ", overall image loss: {:.3f}".format(overall_reconstruction_loss.mean(dim=-1).sum(dim=-1).mean())
    to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - start))
    print(to_print)
    start = time.time()
