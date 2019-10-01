import torch
from torch.utils.data import DataLoader

from networks import *
from utils import *
from visdom_utils import *
from reinforce import *
from dynamicmap import DynamicMap

import time
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
np.set_printoptions(precision=3)

# args:
BATCH_SIZE = 4
SEED = 123
START_SEQ_LEN = 25
END_SEQ_LEN = 25
ATTN_SIZE = 5
# ENV_SIZE = 16
# MAP_SIZE = 8
ATTN_SIZE = 21
ENV_SIZE = 84
MAP_SIZE = 21
MAP_CHANNELS = 64

# env_name = 'DynamicObjects-v1'
# from dynamicobject import DynamicObjects
# env = DynamicObjects(size=ENV_SIZE)
# CHANNELS = env.observation_space.shape[2]
# env_name = 'PhysEnv'
# env_name = 'Breakout-v4'
env_name = 'Pong-v4'
CHANNELS = 3

np.random.seed(SEED)
torch.manual_seed(SEED)

# initialize training data
demo_dir = '/home/himanshu/experiments/DynamicNeuralMap/trainingdata-{}/'.format(env_name)
print('using training data from {}'.format(demo_dir))
dataset = CurriculumDataset(demo_dir=demo_dir, preload=False)
seq_len = START_SEQ_LEN
dataset.set_seqlen(seq_len)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                          num_workers=8, collate_fn=time_collate,
                          drop_last=True, pin_memory=True)

# gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("will run on {} device!".format(device))

# initialize map
map = DynamicMap(
    size=MAP_SIZE,
    channels=MAP_CHANNELS,
    env_size=ENV_SIZE,
    env_channels=CHANNELS,
    batchsize=BATCH_SIZE,
    device=device)
map.to(device)

policy_network = PolicyFunction_21_84(channels=MAP_CHANNELS)
# policy_network = PolicyFunction_x_x(channels=MAP_CHANNELS)
value_network = ValueFunction(channels=MAP_CHANNELS, input_size=MAP_SIZE)
glimpse_agent = A2CPolicy(
    output_size=ENV_SIZE,
    policy_network=policy_network,
    value_network=value_network,
    device=device)

mse = MSEMasked()
mse_unmasked = nn.MSELoss()
optimizer = optim.Adam(map.parameters(), lr=1e-4)

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


# iterate through data and learn!
training_metrics = OrderedDict([
    ('write cost', AverageMeter()),
    ('step cost', AverageMeter()),
    ('post write', AverageMeter()),
    ('post step', AverageMeter()),
    ('overall', AverageMeter()),
    ('policy_loss', AverageMeter()),
    ('policy_entropy', AverageMeter()),
    ('val_loss', AverageMeter()),
    ('avg_val', AverageMeter()),
    ('avg_reward', AverageMeter()),
    ])

i_batch = 0
start = time.time()
for epoch in range(10000):
    train_loader_iter = iter(train_loader)
    for batch in train_loader_iter:
        state_batch = batch
        state_batch = state_batch.to(device)
        # initialize map with initial input glimpses
        map.reset()
        optimizer.zero_grad()
        # get an empty reconstruction
        post_step_reconstruction = map.reconstruct()
        # pick starting locations of attention (random)
        loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2), random=False)
        loc = np.clip(loc, ATTN_SIZE//2, ENV_SIZE - 1 - ATTN_SIZE//2).astype(np.int64)  # clip to avoid edges
        # get started training!
        attn_log_probs = []
        attn_rewards = []
        total_write_loss = 0
        total_step_loss = 0
        total_post_write_loss = 0
        total_post_step_loss = 0
        overall_reconstruction_loss = 0
        # gather initial glimpse from state
        obs_mask = create_attn_mask(loc)
        minus_obs_mask = 1-obs_mask
        post_step_loss = mse(post_step_reconstruction, state_batch[0], obs_mask)
        # save the loss as a reward for glimpse agent
        glimpse_agent.reward(post_step_loss.detach())
        loss = 0
        for t in range(1, seq_len):
            post_step_reconstruction = post_step_reconstruction * minus_obs_mask + state_batch[t-1] * obs_mask
            write_cost = map.write(post_step_reconstruction.detach(), obs_mask, minus_obs_mask)
            map.write(post_step_reconstruction.detach(), obs_mask, minus_obs_mask)
            # post-write reconstruction loss
            post_write_reconstruction = map.reconstruct()
            post_write_loss = mse(post_write_reconstruction, state_batch[t-1], obs_mask).mean()
            # step forward the internal map
            step_cost = map.step()
            post_step_reconstruction = map.reconstruct()
            # select next attention spot
            loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2), random=False)
            loc = np.clip(loc, ATTN_SIZE//2, ENV_SIZE - 1 - ATTN_SIZE//2).astype(np.int64)  # clip to avoid edges
            next_obs_mask = create_attn_mask(loc)
            # compute post-step reconstruction loss
            post_step_loss = mse(post_step_reconstruction, state_batch[t], next_obs_mask)
            # save the loss as a reward for glimpse agent
            glimpse_agent.reward(post_step_loss.detach())
            post_step_loss = post_step_loss.mean()
            # add up all losses
            loss += 0.01 * (write_cost + step_cost) + post_write_loss + post_step_loss
            total_write_loss += 0.01 * write_cost.item()
            total_step_loss += 0.01 * + step_cost.item()
            total_post_write_loss += post_write_loss.item()
            total_post_step_loss += post_step_loss.item()
            overall_reconstruction_loss += mse_unmasked(state_batch[t], post_step_reconstruction).item()
            obs_mask = next_obs_mask
            minus_obs_mask = 1-obs_mask
        # finally propagate loss back through entire training sequence
        loss.backward()
        optimizer.step()
        glimpse_agent.update(training_metrics, skip_train=False)
        training_metrics['write cost'].update(total_write_loss)
        training_metrics['step cost'].update(total_step_loss)
        training_metrics['post write'].update(total_post_write_loss)
        training_metrics['post step'].update(total_post_step_loss)
        training_metrics['overall'].update(overall_reconstruction_loss)
        i_batch += 1

        if i_batch % 100 == 0:
            to_print = 'epoch [{}] batch [{}]'.format(epoch, i_batch)
            for key, value in training_metrics.items():
                if type(value) == AverageMeter:
                    to_print += ", {}: {:.3f}".format(key, value.avg)
            to_print += ", time/it (ms): {:.3f}".format(1000 * (time.time() - start)/100)
            print(to_print)
            start = time.time()
        if i_batch % 1000 == 0:
            agentsavepath = os.path.join('/home/himanshu/experiments/DynamicNeuralMap', env_name, '21map_faithfulnetwork_residualstep')
            print('saving network weights to {} ...'.format(agentsavepath))
            map.save(os.path.join(agentsavepath, 'map{}.pth'.format(i_batch)))
            glimpse_net = {'policy_network': policy_network, 'value_network': value_network}
            torch.save(glimpse_net, os.path.join(agentsavepath, 'glimpsenet{}.pth'.format(i_batch)))
        if i_batch > 5000 and i_batch % 2000 == 0:
            if seq_len < END_SEQ_LEN:
                seq_len += 1
                print("INCREASING training sequence length to {}".format(seq_len))
                dataset.set_seqlen(seq_len)
                train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                          num_workers=8, collate_fn=time_collate,
                                          drop_last=True, pin_memory=True)
                break
