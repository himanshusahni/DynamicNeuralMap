import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks import *
from utils import SequenceDataset, MSEMasked, save_example_images
from rl import *
from dynamicmap import DynamicMap

import time
import numpy as np
import os
np.set_printoptions(precision=3)

# args:
BATCH_SIZE = 4
SEED = 123
ATTN_SIZE = 5
# ENV_SIZE = 16
# MAP_SIZE = 8
ATTN_SIZE = 21
ENV_SIZE = 84
MAP_SIZE = 21
MAP_CHANNELS = 48

# env_name = 'DynamicObjects-v1'
# from dynamicobject import DynamicObjects
# env = DynamicObjects(size=ENV_SIZE)
# CHANNELS = env.observation_space.shape[2]
env_name = 'PhysEnv'
# env_name = 'Breakout-v4'
# env_name = 'Pong-v4'
CHANNELS = 3
class ENV:
    def render(self, img):
        img = (img + 1)/2.
        img = img * 255
        img = img.astype(np.uint8)
        return img
env = ENV()

np.random.seed(SEED)
torch.manual_seed(SEED)

# initialize training data
demo_dir = '/home/himanshu/experiments/DynamicNeuralMap/testingdata-{}-random/'.format(env_name)
print('using training data from {}'.format(demo_dir))
dataset = SequenceDataset(data_dir=demo_dir)
print("created dataset")
seq_len = 25
dataset.seq_len = seq_len
test_idxs = list(range(len(dataset)))
test_idx_sampler = SubsetRandomSampler(test_idxs)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=test_idx_sampler, num_workers=1, collate_fn=dataset.time_collate,
                         drop_last=True)
test_loader_iter = iter(test_loader)

# gpu?
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("will run on {} device!".format(device))

# initialize map
map = DynamicMap(
    size=MAP_SIZE,
    channels=MAP_CHANNELS,
    env_size=ENV_SIZE,
    env_channels=CHANNELS,
    nb_actions=4,
    batchsize=BATCH_SIZE,
    device=device)
map.to(device)
mse = MSEMasked()
mse_unmasked = nn.MSELoss(reduction='none')
# saved models
name = '21map_DMM_offpolicyglimpse'
model_dir = '/home/himanshu/experiments/DynamicNeuralMap/{}/{}/'.format(env_name, name)
model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir)
               if name.endswith('pth') and
               # '91000' in name and
               'map' in name]

attn_span = range(-(ATTN_SIZE//2), ATTN_SIZE//2+1)
xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)

idxs_dim_0 = np.repeat(np.arange(BATCH_SIZE), ATTN_SIZE * ATTN_SIZE)
def create_attn_mask(loc):
    """create a batched mask out of batched attention locations"""
    attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
    idxs_dim_2 = attn[:, 0, :].flatten()
    idxs_dim_3 = attn[:, 1, :].flatten()
    obs_mask = torch.zeros(BATCH_SIZE, 1, ENV_SIZE, ENV_SIZE)
    obs_mask[idxs_dim_0, :, idxs_dim_2, idxs_dim_3] = 1
    obs_mask = obs_mask.to(device)
    return obs_mask

start = time.time()
for path in model_paths:
    # load the model
    print("loading " + path)
    map.load(path)
    map.to(device)
    it = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1][3:])
    # load the glimpse agent
    pathdir, pathname = os.path.split(path)
    glimpsenet = torch.load(os.path.join(pathdir, pathname.replace("map", "glimpsenet")), map_location='cpu')
    glimpse_q = glimpsenet['q'].to(device)
    glimpse_agent = OffPolicyGlimpseAgent(
        output_size=ENV_SIZE,
        attn_size=ATTN_SIZE,
        batchsize=BATCH_SIZE,
        q_arch=PolicyFunction_21_84,
        channels=MAP_CHANNELS,
        device=device, )
    glimpse_agent.dqn.q = glimpse_q
    glimpse_agent.pi = glimpse_q
    # draw a testing batch
    try:
        test_batch = next(test_loader_iter)
    except StopIteration:
        test_loader_iter = iter(test_loader)
        test_batch = next(test_loader_iter)
    state_batch, unmasked_state_batch, action_batch, glimpse_action_batch, reward_batch = test_batch
    # send to gpu
    unmasked_state_batch = unmasked_state_batch.to(device)
    action_batch = action_batch.to(device)
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
    map.reset()
    # get an empty reconstruction
    post_step_reconstruction = map.reconstruct()
    loc = glimpse_agent.step(map.map.detach(), random=False)
    logits = glimpse_agent.pi(map.map.detach().to(device))
    test_maps_heatmaps.append(F.softmax(logits[0], dim=-1).view(1, ENV_SIZE, ENV_SIZE).detach().cpu())
    loc = np.clip(loc, ATTN_SIZE // 2, ENV_SIZE - 1 - ATTN_SIZE // 2).astype(np.int64)  # clip to avoid edges
    test_locs.append(loc[0].copy())
    obs_mask = create_attn_mask(loc)
    minus_obs_mask = 1-obs_mask
    # get an empty reconstruction
    post_step_reconstruction = map.reconstruct()
    for t in range(seq_len):
        # compute reconstruction loss
        post_step_loss.append(mse(post_step_reconstruction, unmasked_state_batch[t], obs_mask).detach().cpu().numpy())
        # calculate per-channel losses on overall image
        channel_loss = []
        for ch in range(CHANNELS):
            l = mse_unmasked(post_step_reconstruction[:, ch].flatten(start_dim=1), unmasked_state_batch[t][:, ch].flatten(start_dim=1))
            l = l.mean(dim=1)
            channel_loss.append(l.detach().cpu().numpy())
        overall_reconstruction_loss.append(channel_loss)
        # obs = state_batch[t] * obs_mask + post_step_reconstruction.detach() * minus_obs_mask
        obs = unmasked_state_batch[t] * obs_mask
        write_loss += map.write(obs, obs_mask, minus_obs_mask)
        post_write_reconstruction = map.reconstruct()
        post_write_loss.append(mse(post_write_reconstruction, unmasked_state_batch[t-1], obs_mask).detach().cpu().numpy())
        test_maps_prestep.append(post_write_reconstruction[0].detach().cpu())
        # step forward the internal map
        # onehot_action = torch.zeros(BATCH_SIZE, 4).to(device)
        # onehot_action[:, t%4] = 1
        actions = action_batch[t]
        actions = actions.unsqueeze(dim=1)
        onehot_action = torch.zeros(BATCH_SIZE, 4).to(device)
        onehot_action.scatter_(1, actions, 1)
        step_cost = map.step(onehot_action)
        # step_cost = map.step(F.softmax(torch.rand(BATCH_SIZE, 4), dim=0).to(device))
        post_step_reconstruction = map.reconstruct()
        test_maps_poststep.append(post_step_reconstruction[0].detach().cpu())
        # select next attention spot
        loc = glimpse_agent.step(map.map.detach(), random=False)
        logits = glimpse_agent.pi(map.map.detach().to(device))
        test_maps_heatmaps.append(F.softmax(logits[0], dim=-1).view(1, ENV_SIZE, ENV_SIZE).detach().cpu())
        sigma.append(glimpse_agent.policy.entropy(logits).detach().cpu().numpy())
        loc = np.clip(loc, ATTN_SIZE//2, ENV_SIZE - 1 - ATTN_SIZE//2).astype(np.int64)  # clip to avoid edges
        test_locs.append(loc[0].copy())
        obs_mask = create_attn_mask(loc)
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
    }, os.path.join(model_dir, 'loss_{}.pt'.format(model_name)))
    # save some generated images
    save_example_images(
       [unmasked_state_batch[t][0].cpu() for t in range(seq_len)],
       test_maps_heatmaps,
       test_maps_prestep,
       test_maps_poststep,
       test_locs,
       os.path.join(model_dir, 'predictions_{}.jpeg'.format(model_name)),
       env)
    to_print = "[{}] test loss: {:.3f}".format(model_name, test_loss)
    to_print += ", overall image loss: {:.3f}".format(overall_reconstruction_loss.mean())
    to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - start))
    print(to_print)
    start = time.time()
