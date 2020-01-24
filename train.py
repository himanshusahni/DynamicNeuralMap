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
from utils import SequenceDataset, MSEMasked
from rl import OffPolicyGlimpseAgent
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

    # env_name = 'DynamicObjects-v1'
    # from dynamicobject import DynamicObjects
    # env = DynamicObjects(size=ENV_SIZE)
    # CHANNELS = env.observation_space.shape[2]
    env_name = 'PhysEnv'
    # env_name = 'Breakout-v4'
    # env_name = 'Pong-v4'
    CHANNELS = 3
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # initialize training data
    demo_dir = '/home/himanshu/experiments/DynamicNeuralMap/data-{}-random/'.format(env_name)
    print('using training data from {}'.format(demo_dir))
    dataset = SequenceDataset(data_dir=demo_dir)

    # gpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    glimpse_agent = OffPolicyGlimpseAgent(
        output_size=ENV_SIZE,
        attn_size=ATTN_SIZE,
        batchsize=BATCH_SIZE,
        q_arch=PolicyFunction_21_84,
        channels=MAP_CHANNELS,
        device=device,)
    optimizer = optim.Adam(map.parameters(), lr=1e-4)

    # iterate through data and learn!
    training_metrics = OrderedDict([
        ('map/write_cost', AverageMeter(10)),
        ('map/step_cost', AverageMeter(10)),
        ('map/post_write', AverageMeter(10)),
        ('map/post_step', AverageMeter(10)),
        ('map/overall', AverageMeter(10)),
        ('map/min_overall', AverageMeter(10)),
        ('glimpse/loss', AverageMeter(10)),
        ('glimpse/policy_entropy', AverageMeter(10)),
        ('glimpse/avg_val', AverageMeter(10)),
        ('glimpse/avg_reward', AverageMeter(10)),
        ])

    i_batch = 0
    start = time.time()
    for epoch in range(10000):
        # train_loader_iter = iter(train_loader)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                  num_workers=8, collate_fn=dataset.time_collate,
                                  drop_last=True, pin_memory=True)
        for (state_batch, unmasked_state_batch, action_batch, glimpse_action_batch, reward_batch) in train_loader:
            state_batch = state_batch.to(device)
            unmasked_state_batch = unmasked_state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            mask_batch = []
            for t in range(glimpse_action_batch.size(0)):
                glimpse_action_clipped = glimpse_agent.norm_and_clip(glimpse_action_batch[t])
                mask_batch.append(glimpse_agent.create_attn_mask(glimpse_action_clipped)[0])
            mask_batch = torch.cat([mask.unsqueeze(0) for mask in mask_batch], dim=0)
            glimpse_action_batch = glimpse_action_batch.to(device)
            # get started training!
            optimizer.zero_grad()
            loss = map.lossbatch(
                state_batch,
                action_batch,
                reward_batch,
                glimpse_agent,
                training_metrics,
                mask_batch=mask_batch,
                unmasked_state_batch=unmasked_state_batch,
                glimpse_action_batch=glimpse_action_batch,
            )
            # propagate loss back through entire training sequence
            loss.backward()
            optimizer.step()
            # and update the glimpse agent
            glimpse_agent.update(map.map.detach(), 1, training_metrics, scope='glimpse')
            glimpse_agent.reset()
            i_batch += 1

            if i_batch % 10 == 0:
                to_print = 'epoch [{}] batch [{}]'.format(epoch, i_batch)
                for key, value in training_metrics.items():
                    if type(value) == AverageMeter:
                        to_print += ", {}: {:.3f}".format(key, value.avg)
                to_print += ", time/it (ms): {:.3f}".format(1000 * (time.time() - start)/100)
                print(to_print)
                start = time.time()
            if i_batch % 1000 == 0:
                agentsavepath = os.path.join('/home/himanshu/experiments/DynamicNeuralMap', env_name, '21map_DMM_offpolicyglimpse')
                print('saving network weights to {} ...'.format(agentsavepath))
                map.save(os.path.join(agentsavepath, 'map{}.pth'.format(i_batch)))
                # glimpse_net = glimpse_agent.ppo.actor_critic
                glimpse_net = {'q': glimpse_agent.dqn.q,}
                torch.save(glimpse_net, os.path.join(agentsavepath, 'glimpsenet{}.pth'.format(i_batch)))
            # if i_batch > 5000 and i_batch % 2000 == 0:
            #     if seq_len < END_SEQ_LEN:
            #         seq_len += 1
            #         print("INCREASING training sequence length to {}".format(seq_len))
            #         dataset.set_seqlen(seq_len)
            #         train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
            #                                   num_workers=8, collate_fn=time_collate,
            #                                   drop_last=True, pin_memory=True)
            #         break
