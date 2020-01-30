import time
import numpy as np
from collections import OrderedDict
import os
import matplotlib
matplotlib.use('Agg')
np.set_printoptions(precision=3)

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorch_rl.utils import AverageMeter
from utils import SequenceDataset, time_collate, MSEMasked
from rl import GlimpseAgent
from networks import *
from dynamicmap import DynamicMap


if __name__ == "__main__":
    # args:
    BATCH_SIZE = 8
    SEED = 123
    START_SEQ_LEN = 25
    END_SEQ_LEN = 25

    ATTN_SIZE = 21
    ENV_SIZE = 84
    MAP_SIZE = 21
    MAP_CHANNELS = 48

    env_name = 'PhysEnv'
    CHANNELS = 3
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # initialize training data
    demo_dir = '/home/himanshu/experiments/DynamicNeuralMap/trainingdata-{}-v1/'.format(env_name)
    print('using training data from {}'.format(demo_dir))
    dataset = SequenceDataset(data_dir=demo_dir)
    seq_len = START_SEQ_LEN
    dataset.set_seqlen(seq_len)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              num_workers=8, collate_fn=time_collate,
                              drop_last=True, pin_memory=True)

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

    policy_network = PolicyFunction_21_84(channels=MAP_CHANNELS)
    value_network = ValueFunction(channels=MAP_CHANNELS, input_size=MAP_SIZE)
    glimpse_agent = GlimpseAgent(
        output_size=ENV_SIZE,
        attn_size=ATTN_SIZE,
        batchsize=BATCH_SIZE,
        policy_network=policy_network,
        value_network=value_network,
        device=device,)

    optimizer = optim.Adam(map.parameters(), lr=1e-4)

    # iterate through data and learn!
    training_metrics = OrderedDict([
        ('map/write_cost', AverageMeter()),
        ('map/step_cost', AverageMeter()),
        ('map/post_write', AverageMeter()),
        ('map/post_step', AverageMeter()),
        ('map/overall', AverageMeter()),
        ('map/min_overall', AverageMeter()),
        ('glimpse/policy_loss', AverageMeter()),
        ('glimpse/policy_entropy', AverageMeter()),
        ('glimpse/val_loss', AverageMeter()),
        ('glimpse/avg_val', AverageMeter()),
        ('glimpse/avg_reward', AverageMeter()),
        ])

    i_batch = 0
    start = time.time()
    for epoch in range(10000):
        train_loader_iter = iter(train_loader)
        for (state_batch, action_batch, reward_batch) in train_loader_iter:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            # get started training!
            optimizer.zero_grad()
            loss = map.lossbatch(
                state_batch,
                action_batch,
                reward_batch,
                glimpse_agent,
                training_metrics)
            # propagate loss back through entire training sequence
            loss.backward()
            optimizer.step()
            # and update the glimpse agent
            glimpse_agent.update(map.map.detach(), None, training_metrics, None, scope='glimpse')
            glimpse_agent.reset()
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
                agentsavepath = os.path.join('/home/himanshu/experiments/DynamicNeuralMap', env_name, '21map_DMM_actioncondition2')
                print('saving network weights to {} ...'.format(agentsavepath))
                map.save(os.path.join(agentsavepath, 'map{}.pth'.format(i_batch)))
                # glimpse_net = glimpse_agent.ppo.actor_critic
                glimpse_net = {'policy_network': glimpse_agent.a2c.pi,
                               'value_network': glimpse_agent.a2c.V}
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
