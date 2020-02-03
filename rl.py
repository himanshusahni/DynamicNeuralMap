import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical, MultivariateNormal

from pytorch_rl import policies, algorithms, agents
from pytorch_rl.utils import AverageMeter

from dynamicmap import DynamicMap, StackMemory
from networks import *

import numpy as np
from collections import deque


class GlimpseAgent():
    """wrapper around an A2C agent for glimpse actions"""
    def __init__(self, output_size, attn_size, batchsize, policy_network, value_network, device):
        self.output_size = output_size
        self.attn_size = attn_size
        self.batchsize = batchsize
        self.policy_network = policy_network
        self.value_network = value_network
        self.policy = policies.MultinomialPolicy()
        self.device = device
        self.a2c = algorithms.A2C(policy_network, value_network, self.policy, device, gamma=0.9,
                                  entropy_weighting=.01)
        self.pi = self.a2c.pi

        attn_span = range(-(self.attn_size// 2), self.attn_size// 2 + 1)
        self.xy = np.flip(np.array(np.meshgrid(attn_span, attn_span)), axis=0).reshape(2, -1)
        self.idxs_dim_0 = np.repeat(np.arange(self.batchsize), self.attn_size * self.attn_size)

        self.reset()

    def step(self, x, random=False, test=False):
        """predict action based on input and update internal storage"""
        self.states.append(x)
        if random:
            action = np.random.randint(0, self.output_size * self.output_size, size=(x.size(0),))
        else:
            logits = self.pi(x)
            self.logits.append(logits)
            action = self.policy(logits, test)
            self.actions.append(action)
            action = action.detach().cpu().numpy()
        # normalize actions to environment range
        action = self.norm_and_clip(action)
        return action

    def norm_and_clip(self, action):
        """
        convert action to environment range and clip according to attention size
        :param action:
        :return:
        """
        action = np.unravel_index(action, (self.output_size, self.output_size))
        if type(action[0]) == np.ndarray:
            action = np.array(list(zip(*action)))
        else:
            action = np.array(action)
        # clip to avoid edges
        action = np.clip(action, self.attn_size // 2, self.output_size - 1 - self.attn_size// 2).astype(np.int64)
        return action

    def reward(self, r, d=None):
        """reward glimpse agent for most recent prediction"""
        self.rewards.append(r)
        if d is None:
            self.dones.append(torch.zeros((r.size(0),)).to(self.device))
        else:
            self.dones.append(d)

    def update(self, final_state, dones, metrics, old_logits=None, scope=''):
        """update glimpse policy using experience from this episode"""
        # designate last state of sequence as terminal for glimpse agent
        self.dones[-1][:] = 1.
        states = torch.cat([state.unsqueeze(dim=1) for state in self.states], dim=1)
        actions = torch.cat([action.unsqueeze(dim=1) for action in self.actions], dim=1)
        rewards = torch.cat([reward.unsqueeze(dim=1) for reward in self.rewards], dim=1)
        dones = torch.cat([done.unsqueeze(dim=1) for done in self.dones], dim=1)
        exp = (states, actions, rewards, dones)
        loss = self.a2c.update(exp, final_state, True, metrics, old_logits, scope)

    def create_attn_mask(self, loc):
        """create a batched mask out of batched attention locations"""
        attn = loc[range(self.batchsize), :, np.newaxis] + self.xy  # get all indices in attention window size
        idxs_dim_2 = attn[:, 0, :].flatten()
        idxs_dim_3 = attn[:, 1, :].flatten()
        obs_mask = torch.zeros(self.batchsize, 1, self.output_size, self.output_size)
        obs_mask[self.idxs_dim_0, :, idxs_dim_2, idxs_dim_3] = 1
        obs_mask = obs_mask.to(self.device)
        return obs_mask, 1-obs_mask

    def reset(self):
        """resets the internal tracking variables"""
        self.states = []
        self.logits = []
        self.actions = []
        self.rewards = []
        self.dones = []


class AttentionConstrainedEnvironment:
    """
    observations from underlying environments are constrained by a
    controllable hard attention mechanism.
    Can think of observations in the environment as tuples of masked environment observations and the observation
    mask. Can think of actions in this environment as actions in the underlying environment as well as changing
    glimpse location for next step.
    """
    def __init__(self, env_size, attn_size, device):
        self.env_size = env_size
        self.attn_size = attn_size
        self.device = device

    def preprocess(self, img):
        """
        converts environment provided observations to renormalized pytorch tensor
        :param img: [0, 255] uint8 numpy array.
        :return: [-1, 1] pytorch tensor
        """
        img = img.transpose((2, 0, 1))  # pytorch expects CxHxW
        img = torch.from_numpy(img).float()
        img = img / 127.5 - 1  # renormalize
        return img.to(self.device)

    def mask_state(self, state, loc):
        """
        creates a partially observed state from the full environment state.
        A mask is created around the location provided according to the
        attention window size and then multiplied with the state
        :param state: full environment observation (CxHxW pytorch tensor)
        :param loc: (x, y) tuple location of attention window
        :return: (masked state (glimpse), observation mask)
        """
        # construct mask
        obs_mask = torch.zeros(1, self.env_size, self.env_size)
        obs_mask[:,
        loc[0] - self.attn_size//2 : loc[0] + self.attn_size//2 + 1,
        loc[1] - self.attn_size//2 : loc[1] + self.attn_size//2 + 1] = 1
        obs_mask = obs_mask.to(self.device)
        glimpse = state * obs_mask
        return glimpse, state, obs_mask

    def reset(self, loc=None):
        """
        starts a new episode
        :param loc: location of first glimpse, if None, chosen randomly
        :return: first observation of episode and its associated mask
        """
        self.ep_step = 0
        # first reset the underlying environment and get a state
        state = self.env.reset()
        state = self.preprocess(state)
        if loc is None:
            # pick a random location
            loc = np.random.randint(0, self.env_size, size=(2,))
            # clip to avoid edges
            loc = np.clip(loc, self.attn_size // 2, self.env_size - 1 - self.attn_size// 2).astype(np.int64)
        return self.mask_state(state, loc)

    def step(self, action, loc=None):
        """
        Take an action in the underlying environment and change glimpse location for next state
        :param action: action to be taken in underlying environment that produces next observation and reward
        :param loc: glimpse location for the next observation.
        :return: a tuple of masked observation at the specified location and its mask, along with the usual
        reward, done and info
        """
        self.ep_step += 1
        # step in the environment and get next observation as usual
        state, r, done, _ = self.env.step(action)
        state = self.preprocess(state)
        if loc is None:
            # pick a random location
            loc = np.random.randint(0, self.env_size, size=(2,))
            # clip to avoid edges
            loc = np.clip(loc, self.attn_size // 2, self.env_size - 1 - self.attn_size// 2).astype(np.int64)
        return self.mask_state(state, loc), r, done, _


class DMMAgent():
    def __init__(self, algorithm, policy, memory_mode, nb_threads, nb_rollout_steps,
                 max_env_steps, state_shape, test_freq, obs_shape, frame_stack,
                 attn_size, batchsize, max_buffer_len, agent_train_delay,
                 device, callbacks):
        """
        Multithreaded on policy discrete agent that uses the DMM architecture to store and update
        its state information. Arguments in addition to vanilla MultithreadedOnPolicyDiscreteAgent
        from pytorch-rl provided below.
        :param obs_shape: Shape of environment observations. Note that this is different from state_shape
        which is shape of state space of agent, which is in this case, the shape of the DMM.
        :param attn_size: size of hard attention window
        :param batchsize: batchsize for DMM training
        :param max_buffer_len: length of buffer storing samples for DMM updates
        :param agent_train_delay: minimum number of samples DMM has been trained on before starting agent training
        """
        self.algorithm = algorithm
        self.policy = policy
        self.memory_mode = memory_mode
        self.nb_threads = nb_threads
        self.nb_rollout_steps = nb_rollout_steps
        self.max_env_steps = max_env_steps
        self.state_shape = state_shape
        self.test_freq = test_freq
        self.obs_shape = obs_shape
        self.frame_stack = frame_stack
        self.batchsize = batchsize
        self.max_buffer_len = max_buffer_len
        self.agent_train_delay = agent_train_delay
        self.device = device
        self.callbacks = callbacks

        self.max_train_steps = int(max_env_steps // (nb_threads * nb_rollout_steps))
        self.dmm_train_delay = self.max_buffer_len * self.nb_threads // 2

        if memory_mode == 'dmm':
            # create DMM
            self.map = DynamicMap(
                size=state_shape[1],
                channels=state_shape[0],
                env_size=obs_shape[1],
                env_channels=obs_shape[0],
                nb_actions=4,
                batchsize=self.batchsize,
                device=device)
        elif memory_mode == 'stack':
            self.map = StackMemory(frame_stack, obs_shape[1], obs_shape[0], batchsize, device)
        self.map.share_memory()
        self.map.to(device)
        self.lr = 1e-4
        self.optimizer = optim.Adam(self.map.parameters(), lr=self.lr)
        # create glimpse agent
        if memory_mode == 'dmm':
            glimpse_policy_network = PolicyFunction_21_84(channels=state_shape[0])
        elif memory_mode == 'stack':
            glimpse_policy_network = PolicyFunction_x_x(channels=state_shape[0])
        glimpse_value_network = ValueFunction(channels=state_shape[0], input_size=state_shape[1])
        glimpse_policy_network.share_memory()
        glimpse_value_network.share_memory()
        self.glimpse_agent = GlimpseAgent(
            output_size=obs_shape[1],
            attn_size=attn_size,
            batchsize=self.batchsize,
            policy_network=glimpse_policy_network,
            value_network=glimpse_value_network,
            device=device)

    class Clone:
        def __init__(self, thread_num, memory_mode, map, glimpse_agent, policy, nb_rollout, rollout, buffer, buffer_len, device):
            """create a new environment"""
            self.t = thread_num
            self.policy = policy
            self.nb_rollout = nb_rollout
            self.rollout = rollout
            self.buffer = buffer
            self.buffer_len = buffer_len
            self.device = device

            # create its own map with copies of the master maps models but batchsize of 1
            if memory_mode == 'dmm':
                self.map = DynamicMap(
                    size=map.size,
                    channels=map.channels,
                    env_size=map.env_size,
                    env_channels=map.env_channels,
                    nb_actions=4,
                    batchsize=1,
                    device=device)
            elif memory_mode == 'stack':
                self.map = StackMemory(map.k, map.size, map.channels, 1, device)
            self.map.assign(map)
            # similarly, create this thread's own glimpse agent
            self.glimpse_agent = GlimpseAgent(
                output_size=glimpse_agent.output_size,
                attn_size=glimpse_agent.attn_size,
                batchsize=1,
                policy_network=glimpse_agent.policy_network,
                value_network=glimpse_agent.value_network,
                device=device)

        def run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            self.env = env
            self.env.env = phys_env.PhysEnv()
            self.buffer_idx = 0
            done = True
            ep_len = 0
            while True:
                global_step = startq.get()
                avg_ep_len = AverageMeter(history=100)
                for step in range(self.nb_rollout):
                    if done:
                        self.map.reset()
                        # starting glimpse location
                        glimpse_state = self.map.content().detach()
                        glimpse_logits = self.glimpse_agent.pi(glimpse_state).detach()
                        glimpse_action = self.glimpse_agent.policy(glimpse_logits).detach()
                        glimpse_action_clipped = self.glimpse_agent.norm_and_clip(glimpse_action.cpu().numpy())
                        obs, unmasked_obs, mask = self.env.reset(loc=glimpse_action_clipped)
                        # write observation to map
                        self.map.write(obs.unsqueeze(dim=0), mask, 1 - mask)
                        state = self.map.content().detach()
                        if ep_len > 0:
                            avg_ep_len.update(ep_len)
                        ep_len = 0
                    if global_step < 10000:
                        # add attention related observations to buffer
                        self.buffer['states'][self.t, self.buffer_idx] = glimpse_state.cpu()
                        self.buffer['logits'][self.t, self.buffer_idx] = glimpse_logits.cpu()
                        self.buffer['obs'][self.t, self.buffer_idx] = obs.cpu()
                        self.buffer['unmasked_obs'][self.t, self.buffer_idx] = unmasked_obs.cpu()
                        self.buffer['masks'][self.t, self.buffer_idx] = mask.cpu()
                        self.buffer['glimpse_actions'][self.t, self.buffer_idx] = glimpse_action.cpu()
                    else:
                        self.buffer = {}
                    # prepare to take a step in the environment!
                    action = self.policy(pi(state)).detach()
                    # step the map forward according to agent action
                    onehot_action = torch.zeros((1, 4)).to(self.device)
                    onehot_action[0, action] = 1
                    self.map.step(onehot_action)
                    # no need to store gradient information for rollouts
                    self.map.detach()
                    # glimpse agent decides where to look after map has stepped
                    glimpse_state = self.map.content().detach()
                    glimpse_logits = self.glimpse_agent.pi(glimpse_state).detach()
                    glimpse_action = self.glimpse_agent.policy(glimpse_logits).detach()
                    glimpse_action_clipped = self.glimpse_agent.norm_and_clip(glimpse_action.cpu().numpy())
                    # step!
                    (next_obs, next_unmasked_obs, next_mask), r, done, _ = \
                        self.env.step(action.cpu().numpy(), loc=glimpse_action_clipped)
                    # add outcomes to rollout buffer for training agent
                    self.rollout['states'][self.t, step] = state
                    self.rollout['actions'][self.t, step] = action
                    self.rollout['rewards'][self.t, step] = r
                    self.rollout['dones'][self.t, step] = float(done)
                    if global_step < 10000:
                        # add some of the info to map buffer as well
                        self.buffer['actions'][self.t, self.buffer_idx] = action.cpu()
                        self.buffer['rewards'][self.t, self.buffer_idx] = r
                        self.buffer['dones'][self.t, self.buffer_idx] = float(done)
                    else:
                        self.buffer = {}
                    # move to next step
                    obs = next_obs
                    unmasked_obs = next_unmasked_obs
                    mask = next_mask
                    # write observation to map
                    self.map.write(obs.unsqueeze(dim=0), mask, 1 - mask)
                    state = self.map.content().detach()
                    self.buffer_idx = (self.buffer_idx + 1) % self.buffer_len
                    ep_len += 1
                # finally add the next state into the states buffer as well to do value estimation
                self.rollout['states'][self.t, self.nb_rollout] = state
                self.rollout['avg_ep_len'][self.t] = avg_ep_len.avg
                stopq.put(self.t)

        def test_run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            self.env = env
            self.env.env = phys_env.PhysEnv()
            while True:
                idx = startq.get()
                self.map.reset()
                # starting glimpse location
                glimpse_logits = self.glimpse_agent.pi(self.map.content().detach())
                glimpse_action = self.glimpse_agent.policy(glimpse_logits).detach()
                glimpse_action = glimpse_action.cpu().numpy()
                glimpse_action_clipped = self.glimpse_agent.norm_and_clip(glimpse_action)
                obs, _, mask = self.env.reset(loc=glimpse_action_clipped)
                ep_reward = 0
                done = False
                while not done:
                    # write observation to map
                    self.map.write(obs.unsqueeze(dim=0), mask, 1-mask)
                    # take a step in the environment!
                    state = self.map.content().detach()
                    action = self.policy(pi(state), test=True).detach()
                    glimpse_logits = self.glimpse_agent.pi(self.map.content().detach())
                    glimpse_action = self.glimpse_agent.policy(glimpse_logits).detach()
                    glimpse_action = glimpse_action.cpu().numpy()
                    glimpse_action_clipped = self.glimpse_agent.norm_and_clip(glimpse_action)
                    (next_obs, _, next_mask), r, done, _ = \
                        self.env.step(action.cpu().numpy(), loc=glimpse_action_clipped)
                    ep_reward += r
                    # step the map forward according to agent action
                    onehot_action = torch.zeros((1, 4)).to(self.device)
                    onehot_action[0, action] = 1
                    self.map.step(onehot_action)
                    # no need to store gradient information for rollouts
                    self.map.detach()
                    # move to next step
                    obs = next_obs
                    mask = next_mask
                print("Testing reward {:.3f}".format(ep_reward))
                self.rollout['test_reward'][idx] = ep_reward
                stopq.put(1)


    def create_batch(self, buffer, samples_added, seqlen, metrics):
        """
        create a batch for training DMM
        :param buffer: multithreaded replay buffer containing glimpses, masks and dones
        :param samples_added: total samples that have been added (same) for each thread
        :param seqlen: length of sequence of states in batch
        :return: minibatch of state and masks both of shape [nb_batch, steps, *state_shape]
        """
        samples_added = samples_added // self.nb_threads  # samples in each row of buffer
        nb_valid_samples = min(samples_added, self.max_buffer_len)
        end_idx = (samples_added - 1) % self.max_buffer_len
        glimpses = []
        masks = []
        unmasked_glimpses = []
        glimpse_states = []
        glimpse_logits = []
        glimpse_actions = []
        actions = []
        rewards = []
        dones = []
        # draw some random numbers
        idxs = np.random.randint(0, nb_valid_samples - seqlen, (self.batchsize,))
        skips = 0
        for idx in idxs :
            while True:
                if samples_added > self.max_buffer_len:
                    seq_idx = range(idx + end_idx + 1, idx + end_idx + 1 + seqlen)
                    seq_idx = [i % self.max_buffer_len for i in seq_idx]
                else:
                    seq_idx = range(idx, idx + seqlen)
                # pick a thread randomly
                t = np.random.randint(0, self.nb_threads)
                if any(buffer['dones'][t, seq_idx[:-1]]):  # last state can be terminal
                    idx = np.random.randint(0, nb_valid_samples - seqlen)
                    skips+=1
                else:
                    break
            glimpses.append(buffer['obs'][t, seq_idx].unsqueeze(0))
            masks.append(buffer['masks'][t, seq_idx].unsqueeze(0))
            unmasked_glimpses.append(buffer['unmasked_obs'][t, seq_idx].unsqueeze(0))
            glimpse_states.append(buffer['states'][t, seq_idx].unsqueeze(0))
            glimpse_logits.append(buffer['logits'][t, seq_idx].unsqueeze(0))
            glimpse_actions.append(buffer['glimpse_actions'][t, seq_idx].unsqueeze(0))
            actions.append(buffer['actions'][t, seq_idx].unsqueeze(0))
            rewards.append(buffer['rewards'][t, seq_idx].unsqueeze(0))
            dones.append(buffer['dones'][t, seq_idx[-1]].unsqueeze(0))
        metrics['map/done_skips'].update(skips)

        glimpses = torch.cat(glimpses, dim=0).transpose(0,1).to(self.device)
        masks = torch.cat(masks, dim=0).transpose(0,1).to(self.device)
        unmasked_glimpses = torch.cat(unmasked_glimpses, dim=0).transpose(0,1).to(self.device)
        glimpse_states = torch.cat(glimpse_states, dim=0).transpose(0,1).to(self.device)
        glimpse_logits = torch.cat(glimpse_logits, dim=0).to(self.device)
        glimpse_actions = torch.cat(glimpse_actions, dim=0).transpose(0,1).to(self.device)
        actions = torch.cat(actions, dim=0).transpose(0,1).to(self.device)
        rewards = torch.cat(rewards, dim=0).transpose(0,1).to(self.device)
        dones = torch.cat(dones, dim=0).to(self.device)
        return glimpses, masks, unmasked_glimpses, glimpse_states, glimpse_logits, glimpse_actions, actions, rewards, dones

    def train(self, make_env):
        """
        main train method for both RL and DMM from experience.
        :param make_env: method for spawning a new environment. Will be passed to multithreaded workers
        """
        # create shared data between agent clones. Current rollout.
        rollout = {
            'states': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps+1,
                *self.state_shape).to(self.device).share_memory_(),
            'actions': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps,
                dtype=torch.long).to(self.device).share_memory_(),
            'rewards': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps).to(self.device).share_memory_(),
            'dones': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps).to(self.device).share_memory_(),
            'avg_ep_len': torch.empty(
                self.nb_threads).share_memory_(),
            'test_reward': torch.empty(self.max_train_steps//self.test_freq).share_memory_()
        }
        # buffer for training DMM
        buffer = {
            'obs': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                *self.obs_shape).share_memory_(),
            'unmasked_obs': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                *self.obs_shape).share_memory_(),
            'states': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                *self.state_shape).share_memory_(),
            'logits': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                self.obs_shape[1]*self.obs_shape[2]).share_memory_(),
            'masks': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                1,
                *self.obs_shape[1:]).share_memory_(),
            'glimpse_actions': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                dtype=torch.long).share_memory_(),
            'actions': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                dtype=torch.long).share_memory_(),
            'rewards': torch.empty(
                self.nb_threads,
                self.max_buffer_len,).share_memory_(),
            'dones': torch.empty(
                self.nb_threads,
                self.max_buffer_len).share_memory_(),
        }
        # stopqs and startqs tell when the actors should collect rollouts
        stopqs = []
        startqs = []
        # make the policy available to all processes
        self.algorithm.actor_critic.share_memory()
        procs = []
        for t in range(self.nb_threads):
            startq = mp.Queue(1)
            startqs.append(startq)
            stopq = mp.Queue(1)
            stopqs.append(stopq)
            c = self.Clone(
                thread_num=t,
                policy=self.policy,
                memory_mode=self.memory_mode,
                map=self.map,
                glimpse_agent=self.glimpse_agent,
                nb_rollout=self.nb_rollout_steps,
                rollout=rollout,
                buffer=buffer,
                buffer_len=self.max_buffer_len,
                device = self.device,
            )
            proc = mp.Process(target=c.run, args=(self.algorithm.pi, make_env(), startq, stopq))
            procs.append(proc)
            proc.start()
        # have a thread for testing
        test_startq = mp.Queue(1)
        test_stopq = mp.Queue(1)
        test_clone = self.Clone(
            thread_num=t+1,
            policy=self.policy,
            memory_mode=self.memory_mode,
            map=self.map,
            glimpse_agent=self.glimpse_agent,
            nb_rollout=self.nb_rollout_steps,
            rollout=rollout,
            buffer=buffer,
            buffer_len=self.max_buffer_len,
            device=self.device,
        )
        test_proc = mp.Process(
            target=test_clone.test_run,
            args=(self.algorithm.pi, make_env(), test_startq, test_stopq))
        test_proc.start()

        # train
        step = 0
        samples_added = 0
        metrics = {'agent/policy_loss': AverageMeter(history=10),
                   'agent/val_loss': AverageMeter(history=10),
                   'agent/policy_entropy': AverageMeter(history=10),
                   'agent/avg_val': AverageMeter(history=10),
                   'agent/avg_reward': AverageMeter(history=10),
                   'agent/avg_ep_len': AverageMeter(history=10),
                   'glimpse/policy_loss': AverageMeter(history=100),
                   'glimpse/val_loss': AverageMeter(history=100),
                   'glimpse/policy_entropy': AverageMeter(history=100),
                   'glimpse/avg_val': AverageMeter(history=100),
                   'glimpse/avg_reward': AverageMeter(history=100),
                   'glimpse/IS_ratio': AverageMeter(history=100),
                   'map/overall': AverageMeter(history=100),
                   'map/min_overall': AverageMeter(history=100),
                   'map/done_skips': AverageMeter(history=100),
                   }
        if self.memory_mode == 'dmm':
            metrics.update({
                'map/write_cost': AverageMeter(history=100),
                'map/step_cost': AverageMeter(history=100),
                'map/post_write': AverageMeter(history=100),
                'map/post_step': AverageMeter(history=100),
            })
        elif self.memory_mode == 'stack':
            metrics['map/masked_loss'] = AverageMeter(history=100)

        while step < self.max_train_steps:
            # start collecting data
            for start in startqs:
                start.put(step)
            # wait for the agents to finish getting data
            for stop in stopqs:
                stop.get()

            samples_added += self.nb_rollout_steps * self.nb_threads
            # update PPO!
            if samples_added > 1.5 * self.dmm_train_delay:
                self.algorithm.update((
                    rollout['states'][:,:-1],
                    rollout['actions'],
                    rollout['rewards'],
                    rollout['dones']), rollout['states'][:,-1], metrics, scope='agent')
            metrics['agent/avg_ep_len'].update(rollout['avg_ep_len'].mean().item())

            # train DMM!
            if samples_added > self.dmm_train_delay:
                if step < 10000:
                    nb_dmm_updates = 15
                else:
                    nb_dmm_updates = 0
                    buffer = {}
                for _ in range(nb_dmm_updates):
                    seqlen = 25
                    glimpses, masks, unmasked_glimpses, glimpse_states, glimpse_logits, glimpse_actions, actions, rewards, dones = \
                        self.create_batch(buffer, samples_added, seqlen, metrics)
                    self.optimizer.zero_grad()
                    loss = self.map.lossbatch(
                        state_batch=glimpses,
                        action_batch=actions,
                        reward_batch=rewards,
                        glimpse_agent=self.glimpse_agent,
                        training_metrics=metrics,
                        mask_batch=masks,
                        unmasked_state_batch=unmasked_glimpses,
                        glimpse_state_batch=glimpse_states,
                        glimpse_action_batch=glimpse_actions)
                    # propagate loss back through entire training sequence
                    loss.backward()
                    self.optimizer.step()
                    # and update the glimpse agent
                    if samples_added > 1.5 * self.dmm_train_delay:
                        # glimpse_logits = None
                        self.glimpse_agent.update(self.map.content().detach(), dones, metrics, glimpse_logits, scope='glimpse')
                    self.glimpse_agent.reset()
            step += 1
            # test
            if step % self.test_freq == 0:
                test_startq.put(step // self.test_freq)
                test_stopq.get()
                metrics['test_reward'] = AverageMeter()
                metrics['test_reward'].update(rollout['test_reward'][step // self.test_freq])
                # test_clone.test_run(self.algorithm.pi, testq)
            # anneal algorithm parameters
            self.algorithm.anneal(step, self.max_train_steps)
            # callbacks
            for callback in self.callbacks:
                callback.on_step_end(step, metrics)
            if step % self.test_freq == 0:
                del metrics['test_reward']

        # end by shutting down processes
        for p in procs:
            p.terminate()
        test_proc.terminate()

    def tosave(self):
        """
        Everything that needs to be saved in order to resuscitate this agent
        :return: A dictionary of dictionary of networks in glimpse agent,
        dynamic map, and ppo algorithm
        """
        return {
            'glimpse': {'policy_network': self.glimpse_agent.policy_network.state_dict(),
                        'value_network': self.glimpse_agent.value_network.state_dict(),
                        'v_optimizer': self.glimpse_agent.a2c.V_optimizer.state_dict(),
                        'pi_optimizer': self.glimpse_agent.a2c.pi_optimizer.state_dict()},
            'actor_critic': {'actor_critic': self.algorithm.actor_critic.state_dict(),
                             'optimizer': self.algorithm.optimizer.state_dict()},
            'map': self.map.tosave(),
            'optimizer': self.optimizer.state_dict()}
