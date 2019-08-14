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
ENV_SIZE = env.observation_space.shape[1]
CHANNELS = env.observation_space.shape[0]
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
# glimpse_net = GlimpseNetwork(nb_actions=2)
# glimpse_net.to(device)
# glimpse_agent = ReinforcePolicyContinuous(glimpse_net, device)
mse = nn.MSELoss()
# saved models
model_dir = '{}'.format(env_name)
model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith('pth') and 'pointread' in name]
#model_paths = [os.path.join(model_dir, name) for name in ['conv_reconstructedwrite_map520000.pth',]]

attn_span = range(-(ATTN_SIZE//2), ATTN_SIZE//2+1)
xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)

start = time.time()
for path in model_paths:
    # load the model
    map.load(path)
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
    all_loc = np.random.rand(seq_len, BATCH_SIZE, 2)  # glimpse location (x,y) in [0,1]
    all_loc = (all_loc*10)  # above in [0, 10]
    all_loc = np.clip(all_loc, 1, 8).astype(np.int64)  # clip to avoid edges
    loc = all_loc[0]
    test_locs = [loc[0].copy()]
    attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
    map.reset(batchsize=BATCH_SIZE)
    # get an empty reconstruction
    reconstruction = map.reconstruct()
    idxs_dim_0 = np.repeat(np.arange(BATCH_SIZE), ATTN_SIZE * ATTN_SIZE)
    idxs_dim_2 = attn[:, 0, :].flatten()
    idxs_dim_3 = attn[:, 1, :].flatten()
    input_glimpses = state_batch[0][idxs_dim_0, :, idxs_dim_2, idxs_dim_3]
    test_maps_prestep = []
    test_maps_poststep = []
    write_loss = 0
    prestep_reconstruction_loss = 0
    poststep_reconstruction_loss = 0
    overall_reconstruction_loss = []
    for t in range(1, seq_len):
        reconstruction[idxs_dim_0, :, idxs_dim_2, idxs_dim_3] = input_glimpses
        write_loss += map.write(reconstruction, (idxs_dim_0, idxs_dim_2, idxs_dim_3)).item()
        reconstruction = map.reconstruct()  # post-write reconstruction
        output_glimpses = reconstruction[idxs_dim_0, :, idxs_dim_2, idxs_dim_3]
        # output_glimpses = output_glimpses.transpose(0, 1).view(-1, BATCH_SIZE, ATTN_SIZE, ATTN_SIZE).transpose(0, 1)
        prestep_reconstruction_loss += mse(output_glimpses, input_glimpses).item()
        test_maps_prestep.append(reconstruction[0].detach().cpu())
        # step forward the internal map
        map.step(action_batch[t-1])
        # select next attention spot
        # loc = np.array([locs[t],]*BATCH_SIZE)
        loc = all_loc[t]
        test_locs.append(loc[0].copy())
        attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
        idxs_dim_0 = np.repeat(np.arange(BATCH_SIZE), ATTN_SIZE * ATTN_SIZE)
        idxs_dim_2 = attn[:, 0, :].flatten()
        idxs_dim_3 = attn[:, 1, :].flatten()
        # now grab next glimpses
        input_glimpses = state_batch[t][idxs_dim_0, :, idxs_dim_2, idxs_dim_3]
        # compute reconstruction loss
        reconstruction = map.reconstruct()
        output_glimpses = reconstruction[idxs_dim_0, :, idxs_dim_2, idxs_dim_3]
        poststep_reconstruction_loss += mse(output_glimpses, input_glimpses).item()
        # calculate per-channel losses on overall image
        channel_loss = []
        for ch in range(CHANNELS):
           channel_loss.append(mse(reconstruction[:, ch], state_batch[t][:, ch]).item())
        overall_reconstruction_loss.append(channel_loss)
        reconstruction = reconstruction.detach()
        test_maps_poststep.append(reconstruction[0].cpu())
    overall_reconstruction_loss = torch.FloatTensor(overall_reconstruction_loss)
    test_loss = 0.01 * write_loss + prestep_reconstruction_loss + poststep_reconstruction_loss
    model_name = os.path.splitext(os.path.basename(path))[0]
    torch.save({
        'write_loss': 0.01 * write_loss,
        'prestep_reconstruction_loss': prestep_reconstruction_loss,
        'poststep_reconstruction_loss': poststep_reconstruction_loss,
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
