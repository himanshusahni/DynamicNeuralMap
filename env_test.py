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
from dynamicmap import MapEnvironment, DynamicMap
from rl import GlimpseAgent
from goalsearch import GoalSearchEnv

from pytorch_rl import utils, callbacks, agents, algorithms, policies

# def preprocess(x):
    # return x

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
                        default=4)
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
                        default=8)
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
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.env = 'PhysEnv'
    def make_env():
        return None

    env = make_env()

    # observation_space = (64, 21, 21)
    observation_space = (3, 84, 84)
    # observation_space = (6, 10, 10)
    if len(observation_space) == 1:
        state_shape = (observation_space[0] * args.frame_stack,)
    else:
        state_shape = (observation_space[0] * args.frame_stack,
                       *observation_space[1:])
    # nb_actions = env.action_space.n
    nb_actions = 4

    # policy_network = PolicyFunction2(args.observation_space[2], args.observation_space[0], args.action_space)
    # value_network = ValueFunction2(args.observation_space[2], args.observation_space[0])

    policy = policies.MultinomialPolicy()
    ppo = algorithms.PPOAC(
        actor_critic_arch=ConvTrunk2,
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
    callbacks = [callbacks.PrintCallback(freq=1),
                 callbacks.SaveNetworks(
                     save_dir='/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_fullyobserved',
                     freq=10,
                     networks={'actor_critic': ppo.actor_critic}),
                 callbacks.SaveMetrics(
                     save_dir='/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_fullyobserved',
                     freq=1000,),
                ]
    agent = agents.MultithreadedOnPolicyDiscreteAgent(
        algorithm=ppo,
        policy=policy,
        nb_rollout_steps=args.nb_rollout_steps,
        state_shape=state_shape,
        max_env_steps=1.01*args.max_train_steps,
        test_freq=args.test_freq,
        nb_threads=args.nb_threads,
        frame_stack=args.frame_stack,
        device=args.device,
        callbacks=callbacks,)
    # preprocess = transforms.Compose([utils.Resize((64, 64)), utils.ImgToTensor()])
    preprocess = utils.ImgToTensor()
    agent.train(make_env, preprocess)

