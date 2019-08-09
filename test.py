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
test_loader = DataLoader(dataset, batch_size=1,
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
model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith('pth') and 'conv' in name and 'step' not in name]
# model_paths = [os.path.join(model_dir, name) for name in ['tanh_map70000.pth']]

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
    #loc = np.random.rand(1, 2)  # glimpse location (x,y) in [0,1]
    #loc = (loc*10)  # above in [0, 10]
    #loc = np.clip(loc, 2, 7).astype(np.int64)  # clip to avoid edges
    step = 0
    loc = np.array([[1+(step%8),1+(step//8)],])
    attn = loc[range(1), :, np.newaxis] + xy  # get all indices in attention window size
    test_loss = 0
    map.reset(batchsize=1)
    input_glimpses = state_batch[0][0, :, attn[0, 0, :], attn[0, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
    map.write(input_glimpses.unsqueeze(dim=0), attn)
    test_maps_prestep = []
    test_maps_poststep = []
    test_locs = [loc.copy()]
    for t in range(1, seq_len):
        test_maps_prestep.append(map.reconstruct().detach().cpu())
        # step forward the internal map
        # map.step(action_batch[t-1])
        # now what does it look like
        reconstruction = map.reconstruct()
        test_maps_poststep.append(reconstruction.detach().cpu())
        # select next attention spot
        #loc = np.random.rand(1, 2)  # glimpse location (x,y) in [0,1]
        #loc = (loc*10)  # above in [0, 10]
        #loc = np.clip(loc, 2, 7).astype(np.int64)  # clip to avoid edges
        step += 1
        loc = np.array([[1+(step%8),1+(step//8)],])
        attn = loc[range(1), :, np.newaxis] + xy  # get all indices in attention window size
        test_locs.append(loc.copy())
        # now grab next glimpses
        input_glimpses = state_batch[t][0, :, attn[0, 0, :], attn[0, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
        # compute reconstruction loss
        output_glimpses = reconstruction[0, :, attn[0, 0, :], attn[0, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
        loss = mse(output_glimpses, input_glimpses)
        test_loss += loss.item()
        map.write(input_glimpses.unsqueeze(dim=0), attn)
    # save some generated images
    model_name = os.path.splitext(os.path.basename(path))[0]
    save_example_images(
        state_batch.cpu(),
        torch.cat(test_maps_prestep, dim=0),
        torch.cat(test_maps_poststep, dim=0),
        np.concatenate(test_locs, axis=0),
        os.path.join(env_name, 'predictions_{}.jpeg'.format(model_name)),
        env)
    to_print = "[{}] test loss: {:.3f}".format(model_name, test_loss)
    to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - start))
    print(to_print)
    start = time.time()
