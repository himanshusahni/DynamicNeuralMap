import argparse

import torch
import torch.multiprocessing as mp

from networks import *
from rl import DMMAgent, AttentionConstrainedEnvironment

from pytorch_rl import utils, callbacks, agents, algorithms, policies, networks

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

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.env = 'PhysEnv'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    observation_space = (48, 21, 21)
    if len(observation_space) == 1:
        state_shape = (observation_space[0] * args.frame_stack,)
    else:
        state_shape = (observation_space[0] * args.frame_stack,
                       *observation_space[1:])
    obs_shape = (3, 84, 84)
    # nb_actions = env.action_space.n
    nb_actions = 4

    # name = '21map_DMM_envdata'
    # name = '21map_DMM_norecurrentbackground_noblend'
    # name = '21map_DMM_actioncondition_taskreward'
    # model_dir = '/home/himanshu/experiments/DynamicNeuralMap/{}/{}/'.format(args.env, name)
    # path = os.path.join(model_dir, 'map200000.pth')
    # map.load(path)
    # map.to(args.device)
    # map.write_model.share_memory()
    # map.step_model.share_memory()
    # path = os.path.join(model_dir, 'glimpsenet200000.pth')
    # glimpsenet = torch.load(path, map_location='cpu')
    # policy_network = glimpsenet['policy_network'].to(args.device)
    # value_network = glimpsenet['value_network'].to(args.device)
    # policy_network = PolicyFunction_21_84(channels=48)
    # value_network = ValueFunction(channels=48, input_size=21)
    # glimpse_agent = GlimpseAgent(
    #     output_size=84,
    #     policy_network=policy_network,
    #     value_network=value_network,
    #     device=args.device)
    # glimpse_pi = policy_network
    # glimpse_pi.share_memory()
    # glimpse_V = value_network
    # glimpse_V.share_memory()
    def make_env():
        return AttentionConstrainedEnvironment(env_size=84, attn_size=21, device=args.device)

    # policy_network = PolicyFunction2(args.observation_space[2], args.observation_space[0], args.action_space)
    # value_network = ValueFunction2(args.observation_space[2], args.observation_space[0])

    policy = policies.MultinomialPolicy()
    ppo = algorithms.PPO(
        actor_critic_arch=networks.ActorCritic,
        trunk_arch=ConvTrunk3,
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
    savedir = '/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_DMM_refactored2'
    calls = [callbacks.PrintCallback(freq=10),
             callbacks.SaveMetrics(
                 save_dir=savedir,
                 freq=1000,),
             ]
    agent = DMMAgent(
        algorithm=ppo,
        policy=policy,
        nb_threads=args.nb_threads,
        nb_rollout_steps=args.nb_rollout_steps,
        max_env_steps=1.01*args.max_train_steps,
        state_shape=state_shape,
        test_freq=args.test_freq,
        frame_stack=args.frame_stack,
        obs_shape=obs_shape,
        attn_size=21,
        batchsize=4,
        max_buffer_len=12500,
        agent_train_delay=30000,
        device=args.device,
        callbacks=calls,)
    agent.callbacks.append(callbacks.SaveNetworks(
        save_dir=savedir,
        freq=100,
        networks=agent.tosave()))
    # finally train
    agent.train(make_env)

