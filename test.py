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

# env = 'Breakout-v4'
env_name = 'Goalsearch-v1'
from goalsearch import GoalSearchEnv
env = GoalSearchEnv(size=10)

# args:
SEED = 354
ATTN_SIZE = 3
ENV_SIZE = env.observation_space.shape[0]
CHANNELS = env.observation_space.shape[2]
BATCH_SIZE = 4

np.random.seed(SEED)
torch.manual_seed(SEED)

# initialize training data
# demo_dir = '../baby-a3c/preprocessed-demos-{}'.format(env)
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
net_trunk = Trunk(ENV_SIZE, 16)
policy_network = PolicyFunction(net_trunk, 2)
value_network = ValueFunction(net_trunk)
mse = MinImposedMSEMasked()
mse_unmasked = MinImposedMSE()
# saved models
model_dir = '{}'.format(env_name)
#model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith('pth') and 'pointread' in name]
model_paths = [os.path.join(model_dir, name) for name in ['conv_a2cattention_map200000.pth',]]

attn_span = range(-(ATTN_SIZE//2), ATTN_SIZE//2+1)
xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)

start = time.time()
idxs_dim_0 = np.repeat(np.arange(BATCH_SIZE), ATTN_SIZE * ATTN_SIZE)
for path in model_paths:
    # load the model
    map.load(path)
    # load the glimpse agent
    glimpsenet = torch.load(path.replace("map", "glimpsenet"))
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
    # locs = np.array(np.meshgrid(range(1,9,1), range(1,9,1))).transpose().reshape(-1,2)
    # loc = np.array([locs[0],]*BATCH_SIZE)
    # loc = np.random.rand(BATCH_SIZE, 2)  # glimpse location (x,y) in [0,1]
    # loc = (loc * ENV_SIZE)  # above in [0, 10]
    # get started training!
    attn_log_probs = []
    attn_rewards = []
    map.reset(batchsize=BATCH_SIZE)
    loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2))
    loc = np.clip(loc, 1, 8).astype(np.int64)  # clip to avoid edges
    attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
    # get an empty reconstruction
    post_step_reconstruction = map.reconstruct()
    idxs_dim_2 = attn[:, 0, :].flatten()
    idxs_dim_3 = attn[:, 1, :].flatten()
    obs_mask = torch.zeros(BATCH_SIZE, 1, ENV_SIZE, ENV_SIZE)
    obs_mask[idxs_dim_0, :, idxs_dim_2, idxs_dim_3] = 1
    obs_mask = obs_mask.to(device)
    minus_obs_mask = 1-obs_mask
    test_maps_prestep = []
    test_maps_poststep = []
    test_locs = [loc[0].copy()]
    write_loss = 0
    post_write_loss = 0
    post_step_loss = 0
    overall_reconstruction_loss = []
    for t in range(1, seq_len):
        post_step_reconstruction = post_step_reconstruction * minus_obs_mask + state_batch[t-1] * obs_mask
        write_loss += map.write(post_step_reconstruction.detach(), obs_mask, minus_obs_mask).item()
        post_write_reconstruction = map.reconstruct()
        post_write_loss += mse(post_write_reconstruction, state_batch[t-1], obs_mask).mean().item()
        test_maps_prestep.append(post_write_reconstruction[0].detach().cpu())
        # step forward the internal map
        map.step(action_batch[t-1])
        # select next attention spot
        # loc = np.array([locs[t],]*BATCH_SIZE)
        loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2))
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
        post_step_reconstruction = map.reconstruct()
        post_step_loss += mse(post_step_reconstruction, state_batch[t], next_obs_mask).mean().item()
        # calculate per-channel losses on overall image
        channel_loss = []
        for ch in range(CHANNELS):
           channel_loss.append(mse_unmasked(post_step_reconstruction[:, ch], state_batch[t][:, ch]).item())
        overall_reconstruction_loss.append(channel_loss)
        test_maps_poststep.append(post_step_reconstruction[0].detach().cpu())
        obs_mask = next_obs_mask
        minus_obs_mask = 1-obs_mask
    overall_reconstruction_loss = torch.FloatTensor(overall_reconstruction_loss)
    test_loss = 0.01 * write_loss + post_write_loss + post_step_loss
    model_name = os.path.splitext(os.path.basename(path))[0]
    torch.save({
        'write_loss': 0.01 * write_loss,
        'post write reconstruction loss': post_write_loss,
        'post step reconstruction loss': post_step_loss,
        'overall_reconstruction_loss': overall_reconstruction_loss,
    }, os.path.join(env_name, 'loss_{}.pt'.format(model_name)))
    # save some generated images
    save_example_images(
        [state_batch[t][0].cpu() for t in range(seq_len)],
        test_maps_prestep,
        test_maps_poststep,
        test_locs,
        os.path.join(env_name, 'predictions_{}.jpeg'.format(model_name)),
        env)
    to_print = "[{}] test loss: {:.3f}".format(model_name, test_loss)
    to_print += ", overall image loss: {:.3f}".format(overall_reconstruction_loss.sum())
    to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - start))
    print(to_print)
    start = time.time()
