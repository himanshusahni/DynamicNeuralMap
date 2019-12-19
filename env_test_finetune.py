import argparse
import numpy as np
import time
import gym
from copy import copy
import matplotlib.pyplot as plt
import os

import torch
import torch.multiprocessing as mp
from torchvision import transforms

from networks import *
from utils import AverageMeter
from dynamicmap import MapEnvironment, DynamicMap, AttentionEnvironment, ConditionalDynamicMap
from rl import GlimpseAgent, DMMRLAgent
from goalsearch import GoalSearchEnv

from pytorch_rl import utils, callbacks, agents, algorithms, policies, networks

ATTN_SIZE = 3
attn_span = range(-(ATTN_SIZE//2), ATTN_SIZE//2+1)
xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)

def preprocess(x):
    return x

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_train_steps',
                        help='maximum environment steps allowed for training',
                        type=int,
                        default=80000000)
    parser.add_argument('--obs_size',
                        help='resize observations from environment',
                        type=int,
                        default=84)
    parser.add_argument('--frame_stack',
                        help='how many observations form a state',
                        type=int,
                        default=1)
    parser.add_argument('--seed',
                        help='random seed',
                        type=int,
                        default=123)
    parser.add_argument('--gamma',
                        help='discount factor',
                        type=float,
                        default=0.99)
    parser.add_argument('--entropy_weighting',
                        help='entropy loss contribution',
                        type=float,
                        default=0.01)
    parser.add_argument('--nb_threads',
                        help='number of processes for collecting experience',
                        type=int,
                        default=4)
    parser.add_argument('--nb_rollout_steps',
                        help='steps per rollout for AC',
                        type=int,
                        default=128)
    parser.add_argument('--test_freq',
                        help='testing frequency',
                        type=int,
                        default=10)
    parser.add_argument('--env',
                        help='environment name',)
    args = parser.parse_args()

    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    args.env = 'PhysEnv'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    observation_space = (64, 21, 21)
    # observation_space = (3, 84, 84)
    # observation_space = (6, 10, 10)
    if len(observation_space) == 1:
        state_shape = (observation_space[0] * args.frame_stack,)
    else:
        state_shape = (observation_space[0] * args.frame_stack,
                       *observation_space[1:])
    obs_shape = (3, 84, 84)
    # nb_actions = env.action_space.n
    nb_actions = 4
    # map = DynamicMap(
    map=ConditionalDynamicMap(
            # map = SpatialNet(
        size=21,
        channels=64,
        env_size=84,
        env_channels=3,
        batchsize=1,
        nb_actions=4,
        device=args.device)

    # name = '21map_DMM_envdata'
    # name = '21map_DMM_norecurrentbackground_noblend'
    name = '21map_DMM_ppoglimpse_actioncondition'
    model_dir = '/home/himanshu/experiments/DynamicNeuralMap/{}/{}/'.format(args.env, name)
    # path = os.path.join(model_dir, 'map160000.pth')
    # path = os.path.join(model_dir, 'map240000.pth')
    path = os.path.join(model_dir, 'map191000.pth')
    map.load(path)
    map.to(args.device)
    map.write_model.share_memory()
    map.step_model.share_memory()
    # path = os.path.join(model_dir, 'glimpsenet160000.pth')
    path = os.path.join(model_dir, 'glimpsenet191000.pth')
    glimpsenet = torch.load(path, map_location='cpu')
    # glimpse_pi = glimpsenet['policy_network'].to(args.device)
    glimpse_pi = glimpsenet.policy_head
    glimpse_pi.share_memory()
    # glimpse_V = glimpsenet['value_network'].to(args.device)
    glimpse_V = glimpsenet.value_head
    glimpse_V.share_memory()
    def make_env():
        glimpse_agent = GlimpseAgent(state_shape, 84, args.device)
        glimpse_agent.load(glimpse_pi, glimpse_V)
        # return GoalSearchEnv(10)
        # return AttentionEnvironment(map, glimpse_pi, 84, 21, args.device)
        return MapEnvironment(map, glimpse_agent, 84, 21, args.device)
        # return None

    # policy_network = PolicyFunction2(args.observation_space[2], args.observation_space[0], args.action_space)
    # value_network = ValueFunction2(args.observation_space[2], args.observation_space[0])

    policy = policies.MultinomialPolicy()
    ppo = algorithms.PPO(
        trunk_arch=ConvTrunk3,
        actor_critic_arch=networks.ActorCritic,
        state_shape=state_shape,
        action_space=nb_actions,
        policy=policy,
        ppo_epochs=4,
        clip_param=0.1,
        target_kl=0.01,
        minibatch_size=256,
        device=args.device,
        gamma=args.gamma,
        lam=0.95,
        clip_value_loss=False,
        value_loss_weighting=0.5,
        entropy_weighting=0.01)
    callbacks = [callbacks.PrintCallback(freq=10),
                 callbacks.SaveNetworks(
                     save_dir='/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_DMM_finetuned_hardcodedglimpse_actioncondition',
                     freq=100,
                     networks={'actor_critic': ppo.actor_critic,
                               'glimpse': {'policy_network': glimpse_pi,
                                           'value_network': glimpse_V},
                               # 'glimpse_actor_critic': glimpsenet,
                               'map': {'write': map.write_model,
                                       'step': map.step_model,
                                       'reconstruct': map.reconstruction_model}}),
                 callbacks.SaveMetrics(
                     save_dir='/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_DMM_finetuned_hardcodedglimpse_actioncondition',
                     freq=1000,),
                 callbacks.ResetMetrics(freq=10),
                ]
    agent = DMMRLAgent(
        algorithm=ppo,
        policy=policy,
        nb_threads=args.nb_threads,
        nb_rollout_steps=args.nb_rollout_steps,
        max_env_steps=1.01*args.max_train_steps,
        state_shape=state_shape,
        obs_shape=obs_shape,
        nb_dmm_updates=5,
        batchsize=4,
        test_freq=args.test_freq,
        frame_stack=args.frame_stack,
        max_buffer_len=10000,
        replay_warmup=1000,
        device=args.device,
        callbacks=callbacks,)

    # preprocess = transforms.Compose([utils.Resize((64, 64)), utils.ImgToTensor()])
    # preprocess = utils.ImgToTensor()
    agent.train(make_env, preprocess)

