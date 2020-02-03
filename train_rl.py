import argparse

import torch
import torch.multiprocessing as mp

from networks import *
from rl import DMMAgent, AttentionConstrainedEnvironment

from pytorch_rl import utils, callbacks, agents, algorithms, policies, networks

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode',
                        help='which memory used',
                        default='dmm')
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

    # state of agent
    if args.mode == 'dmm':
        state_shape = (48, 21, 21)
        trunk = ConvTrunk21
    elif args.mode == 'stack':
        state_shape = (3, 84, 84)
        trunk = ConvTrunk84
    elif args.mode == 'lstm':
        state_shape = (256,)
        trunk = FCTrunk
    if len(state_shape) == 1:
        state_shape = (state_shape[0] * args.frame_stack,)
    else:
        state_shape = (state_shape[0] * args.frame_stack,
                       *state_shape[1:])
    obs_shape = (3, 84, 84)
    nb_actions = 4

    def make_env():
        return AttentionConstrainedEnvironment(env_size=84, attn_size=21, device=args.device)

    policy = policies.MultinomialPolicy()
    savedir = '/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_DMM_refactored_refresh'
    calls = [callbacks.PrintCallback(freq=10),
             callbacks.SaveMetrics(
                 save_dir=savedir,
                 freq=1000,),
             ]
    ppo = algorithms.PPO(
        actor_critic_arch=networks.ActorCritic,
        trunk_arch=trunk,
        state_shape=state_shape,
        action_space=4,
        policy=policy,
        ppo_epochs=4,
        clip_param=0.1,
        target_kl=0.01,
        minibatch_size=256,
        device=args.device,
        gamma=0.99,
        lam=0.95,
        clip_value_loss=False,
        value_loss_weighting=0.5,
        entropy_weighting=0.01)
    agent = DMMAgent(
        algorithm=ppo,
        policy=policy,
        memory_mode=args.mode,
        nb_threads=args.nb_threads,
        nb_rollout_steps=args.nb_rollout_steps,
        max_env_steps=1.01*args.max_train_steps,
        state_shape=state_shape,
        test_freq=args.test_freq,
        obs_shape=obs_shape,
        frame_stack=args.frame_stack,
        attn_size=21,
        batchsize=8,
        max_buffer_len=12500,
        agent_train_delay=30000,
        device=args.device,
        callbacks=calls,)
    agent.callbacks.append(callbacks.SaveNetworks(
        save_dir=savedir,
        freq=100,
        network_func=agent.tosave))
    # finally train
    agent.train(make_env)

