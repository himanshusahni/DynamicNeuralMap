import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks import *
from utils import *
from visdom_utils import *
from reinforce import *
from dynamicmap import DynamicMap

import time
import numpy as np
np.set_printoptions(precision=3)

# env = 'Breakout-v4'
env_name = 'Goalsearch-v0'
from goalsearch import GoalSearchEnv
env = GoalSearchEnv(size=10)

# args:
BATCH_SIZE = 8
SEED = 123
START_SEQ_LEN = 5
END_SEQ_LEN = 15
ATTN_SIZE = 3

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
                          sampler=train_idx_sampler, num_workers=10, collate_fn=time_collate,
                          drop_last=True)
test_loader = DataLoader(dataset, batch_size=1,
                         sampler=test_idx_sampler, num_workers=1, collate_fn=time_collate,
                         drop_last=True)
test_loader_iter = iter(test_loader)

# gpu?
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("will run on {} device!".format(device))

# initialize map
map = DynamicMap(size=10, attn_size=ATTN_SIZE, device=device)
map.to(device)
# map.load('{}/map360000.pth'.format(env_name), device)
# glimpse_net = GlimpseNetwork(nb_actions=2)
# glimpse_net.to(device)
# glimpse_agent = ReinforcePolicyContinuous(glimpse_net, device)
mse = nn.MSELoss()
optimizer = optim.Adam(map.parameters(), lr=1e-6)

attn_span = range(-(ATTN_SIZE//2), ATTN_SIZE//2+1)
xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)

# iterate through data and learn!
training_metrics = {
    'loss': AverageMeter(),}
    # 'policy loss': AverageMeter(),}
i_batch = 0
start = time.time()
for epoch in range(40000):
    train_loader_iter = iter(train_loader)
    for batch in train_loader_iter:
        state_batch, action_batch = batch
        # send to gpu
        state_batch = state_batch.to(device)
        # pick starting locations of attention (random)
        loc = np.random.rand(BATCH_SIZE, 2)  # glimpse location (x,y) in [0,1]
        loc = (loc*10)  # above in [0, 10]
        loc = np.clip(loc, 2, 7).astype(np.int64)  # clip to avoid edges
        attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
        # attn_log_probs = []
        # attn_rewards = []
        # forward pass
        total_loss = 0
        # initialize map with initial input glimpses
        map.reset(BATCH_SIZE)
        # g.render('full.jpeg', state_batch[0][2].permute(1,2,0), loc[2])
        input_glimpses = [state_batch[0][idx, :, attn[idx, 0, :], attn[idx, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
                          for idx in range(BATCH_SIZE)]
        # g.render('glimpse.jpeg', input_glimpses[2].permute(1,2,0), None)
        input_glimpses = torch.cat([g.unsqueeze(dim=0) for g in input_glimpses])
        map.register_glimpse(input_glimpses, attn)
        for t in range(1, seq_len):
            # step forward the internal map
            # map.step(action_batch[t-1])
            # now what does it look like
            reconstruction = map.reconstruct()
            # select next attention spot
            loc = np.random.rand(BATCH_SIZE, 2)  # glimpse location (x,y) in [0,1]
            loc = (loc*10)  # above in [0, 10]
            loc = np.clip(loc, 2, 7).astype(np.int64)  # clip to avoid edges
            attn = loc[range(BATCH_SIZE), :, np.newaxis] + xy  # get all indices in attention window size
            # now grab next glimpses
            input_glimpses = [state_batch[t][idx, :, attn[idx, 0, :], attn[idx, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
                              for idx in range(BATCH_SIZE)]
            input_glimpses = torch.cat([g.unsqueeze(dim=0) for g in input_glimpses])
            # compute reconstruction loss
            output_glimpses = [reconstruction[idx, :, attn[idx, 0, :], attn[idx, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
                               for idx in range(BATCH_SIZE)]
            output_glimpses = torch.cat([g.unsqueeze(dim=0) for g in output_glimpses])
            loss = mse(output_glimpses, input_glimpses)
            # save the loss as a reward for glimpse agent
            # glimpse_agent.reward(loss.detach().cpu().numpy())
            # propogate backwards through entire graph
            loss.backward(retain_graph=True)
            total_loss += loss.item()
            # register the new glimpse
            map.register_glimpse(input_glimpses, attn)
        optimizer.step()
        # policy_loss = glimpse_agent.update()
        training_metrics['loss'].update(total_loss)
        # training_metrics['policy loss'].update(policy_loss)
        i_batch += 1

        ######################
        # Testing and saving #
        ######################
        if i_batch % 1000 == 0:
            # draw a testing batch
            try:
                test_batch = next(test_loader_iter)
            except StopIteration:
                test_loader_iter = iter(test_loader)
                test_batch = next(test_loader_iter)
            state_batch, action_batch = test_batch
            # send to gpu
            state_batch = state_batch.to(device)
            loc = np.random.rand(1, 2)  # glimpse location (x,y) in [0,1]
            loc = (loc*10)  # above in [0, 10]
            loc = np.clip(loc, 2, 7).astype(np.int64)  # clip to avoid edges
            attn = loc[range(1), :, np.newaxis] + xy  # get all indices in attention window size
            test_loss = 0
            map.reset(batchsize=1)
            input_glimpses = state_batch[0][0, :, attn[0, 0, :], attn[0, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
            map.register_glimpse(input_glimpses.unsqueeze(dim=0), attn)
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
                loc = np.random.rand(1, 2)  # glimpse location (x,y) in [0,1]
                loc = (loc*10)  # above in [0, 10]
                loc = np.clip(loc, 2, 7).astype(np.int64)  # clip to avoid edges
                attn = loc[range(1), :, np.newaxis] + xy  # get all indices in attention window size
                test_locs.append(loc.copy())
                # now grab next glimpses
                input_glimpses = state_batch[t][0, :, attn[0, 0, :], attn[0, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
                # compute reconstruction loss
                output_glimpses = reconstruction[0, :, attn[0, 0, :], attn[0, 1, :]].view(-1, ATTN_SIZE, ATTN_SIZE)
                loss = mse(output_glimpses, input_glimpses)
                test_loss += loss.item()
                map.register_glimpse(input_glimpses.unsqueeze(dim=0), attn)
            to_print = 'epoch [{}] batch [{}]'.format(epoch, i_batch)
            for key, value in training_metrics.items():
                if type(value) == AverageMeter:
                    to_print += ", {}: {:.3f}".format(key, value.avg)
            to_print += ", test loss: {:.3f}".format(test_loss)
            to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - start)/1000)
            print(to_print)
            start = time.time()
        if i_batch % 10000 == 0:
            print('saving network weights...')
            map.save(os.path.join(env_name, 'sigmoid_map{}.pth'.format(i_batch)))
        #     torch.save(glimpse_net, os.path.join(env, 'glimpse_net_cont_{}.pth'.format(i_batch)))
            # save some generated images
            save_example_images(
                state_batch.cpu(),
                torch.cat(test_maps_prestep, dim=0),
                torch.cat(test_maps_poststep, dim=0),
                np.concatenate(test_locs, axis=0),
                os.path.join(env_name, 'sigmoid_predictions_map{}.jpeg'.format(i_batch)),
                env)
        if i_batch % 50000 == 0:
            if seq_len < END_SEQ_LEN:
                seq_len += 1
                dataset.set_seqlen(seq_len)
                train_loader_iter = iter(train_loader)
                test_loader_iter = iter(test_loader)
                break
