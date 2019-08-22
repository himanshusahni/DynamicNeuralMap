import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks import *
from utils import *
from visdom_utils import *
from reinforce import *
from dynamicmap import DynamicMap

from pytorch_rl import utils, algorithms, policies

import time
import numpy as np
np.set_printoptions(precision=3)

# env = 'Breakout-v4'
env_name = 'Goalsearch-v1'
from goalsearch import GoalSearchEnv
env = GoalSearchEnv(size=10)

# args:
BATCH_SIZE = 8
SEED = 123
START_SEQ_LEN = 5
END_SEQ_LEN = 25
ATTN_SIZE = 3
ENV_SIZE = env.observation_space.shape[0]
CHANNELS = env.observation_space.shape[2]

np.random.seed(SEED)
torch.manual_seed(SEED)

# initialize training data
# demo_dir = '../baby-a3c/preprocessed-demos-{}'.format(env)
demo_dir = 'data-{}/'.format(env_name)
print('using training data from {}'.format(demo_dir))
dataset = CurriculumDataset(demo_dir=demo_dir, preload=False)
seq_len = START_SEQ_LEN
dataset.set_seqlen(seq_len)
# create train test split on episodes
nb_test_ep = 10
nb_train_ep = len(dataset) - nb_test_ep
print("{} training episodes, {} testing episodes".format(
    nb_train_ep, nb_test_ep))
idxs = list(range(len(dataset)))
np.random.shuffle(idxs)
train_idxs, test_idxs = idxs[:nb_train_ep], idxs[nb_train_ep:]
train_idx_sampler, test_idx_sampler = \
    SubsetRandomSampler(train_idxs), SubsetRandomSampler(test_idxs)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                          sampler=train_idx_sampler, num_workers=8, collate_fn=time_collate,
                          drop_last=True, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=1,
                         sampler=test_idx_sampler, num_workers=1, collate_fn=time_collate,
                         drop_last=True, pin_memory=True)
test_loader_iter = iter(test_loader)

# gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("will run on {} device!".format(device))

# initialize map
map = DynamicMap(size=ENV_SIZE, channels=CHANNELS, attn_size=ATTN_SIZE, device=device)
map.to(device)
# map.load('{}/map360000.pth'.format(env_name), device)

net_trunk = Trunk(ENV_SIZE, 16)
policy_network = PolicyFunction(net_trunk, 2)
value_network = ValueFunction(net_trunk)
glimpse_agent = A2CPolicy(ENV_SIZE, policy_network, value_network, device)
# mse = nn.MSELoss()
mse = MinImposedMSEMasked()
bce = nn.BCELoss()
optimizer = optim.Adam(map.parameters(), lr=1e-6)

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
training_metrics = {
    'loss': AverageMeter(),
    'policy loss': AverageMeter(),}
i_batch = 0
start = time.time()
for epoch in range(10000):
    train_loader_iter = iter(train_loader)
    for batch in train_loader_iter:
        state_batch, action_batch = batch
        state_batch = state_batch.to(device)
        # initialize map with initial input glimpses
        map.reset(BATCH_SIZE)
        # pick starting locations of attention (random)
        loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2))
        loc = np.clip(loc, 1, 8).astype(np.int64)  # clip to avoid edges
        # get started training!
        attn_log_probs = []
        attn_rewards = []
        total_loss = 0
        # get an empty reconstruction
        post_step_reconstruction = map.reconstruct()
        # gather initial glimpse from state
        obs_mask = create_attn_mask(loc)
        minus_obs_mask = 1-obs_mask
        post_step_loss = mse(post_step_reconstruction, state_batch[0], obs_mask)
        # save the loss as a reward for glimpse agent
        glimpse_agent.reward(post_step_loss.detach().cpu())
        for t in range(1, seq_len):
            post_step_reconstruction = post_step_reconstruction * minus_obs_mask + state_batch[t-1] * obs_mask
            write_loss = map.write(post_step_reconstruction.detach(), obs_mask, minus_obs_mask)
            # post-write reconstruction loss
            post_write_reconstruction = map.reconstruct()
            post_write_loss = mse(post_write_reconstruction, state_batch[t-1], obs_mask).mean()
            # step forward the internal map
            map.step(action_batch[t-1])
            # select next attention spot
            loc = glimpse_agent.step(map.map.detach().permute(0, 3, 1, 2))
            loc = np.clip(loc, 1, 8).astype(np.int64)  # clip to avoid edges
            next_obs_mask = create_attn_mask(loc)
            # compute post-step reconstruction loss
            post_step_reconstruction = map.reconstruct()
            post_step_loss = mse(post_step_reconstruction, state_batch[t], next_obs_mask)
            # save the loss as a reward for glimpse agent
            glimpse_agent.reward(post_step_loss.detach().cpu())
            post_step_loss = post_step_loss.mean()
            # propogate backwards through entire graph
            loss = 0.01 * write_loss + post_write_loss + post_step_loss
            loss.backward(retain_graph=True)
            total_loss += loss.item()
            obs_mask = next_obs_mask
            minus_obs_mask = 1-obs_mask
        optimizer.step()
        policy_loss = glimpse_agent.update()
        training_metrics['loss'].update(total_loss)
        training_metrics['policy loss'].update(policy_loss)
        i_batch += 1

        if i_batch % 1000 == 0:
            to_print = 'epoch [{}] batch [{}]'.format(epoch, i_batch)
            for key, value in training_metrics.items():
                if type(value) == AverageMeter:
                    to_print += ", {}: {:.3f}".format(key, value.avg)
            to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - start)/1000)
            print(to_print)
            start = time.time()
        if i_batch % 10000 == 0:
            print('saving network weights...')
            map.save(os.path.join(env_name, 'conv_a2cattention_map{}.pth'.format(i_batch)))
            glimpse_net = {'policy_network': policy_network, 'value_network': value_network}
            torch.save(glimpse_net, os.path.join(env_name, 'conv_a2cattention_glimpsenet{}.pth'.format(i_batch)))
        if i_batch % 20000 == 0:
            if seq_len < END_SEQ_LEN:
                seq_len += 1
                dataset.set_seqlen(seq_len)
                train_loader_iter = iter(train_loader)
                test_loader_iter = iter(test_loader)
                break
