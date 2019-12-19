import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical, MultivariateNormal

from pytorch_rl import policies, algorithms, agents
from pytorch_rl.utils import AverageMeter
from networks import ActorCritic_21_84

import numpy as np
from collections import deque


class GlimpseAgent():
    """uses pytorch-rl ppo"""
    def __init__(self, output_size, policy_network, value_network, device):
        self.output_size = output_size
        self.policy = policies.MultinomialPolicy()
        self.device = device
        self.states = []
        self.logits = []
        self.actions = []
        self.rewards = []
        self.dones = []
        # self.ppo = algorithms.PPO(actor_critic_arch=ActorCritic_21_84,
        #                           trunk_arch=None,
        #                           state_shape=state_shape,
        #                           action_space=None,
        #                           policy=self.policy,
        #                           ppo_epochs=4,
        #                           clip_param=0.1,
        #                           target_kl=0.01,
        #                           minibatch_size=100,
        #                           clip_value_loss=True,
        #                           device=device,
        #                           gamma=0.90,
        #                           lam=0.85,
        #                           value_loss_weighting=0.5,
        #                           entropy_weighting=0.01)
        self.a2c = algorithms.A2C(policy_network, value_network, self.policy, device, gamma=0.9,
                                  entropy_weighting=0.01)
        # self.pi = self.ppo.pi
        self.pi = self.a2c.pi

    # def load(self, policy_network, value_network):
    #     for target_param, param in zip(self.ppo.new_actor_critic.pi_parameters(), policy_network.parameters()):
    #         target_param.data.copy_(param.data)
    #     for target_param, param in zip(self.ppo.new_actor_critic.V_parameters(), value_network.parameters()):
    #         target_param.data.copy_(param.data)
    #     self.ppo.copy_target()
    #
    # def loadAC(self, policy_network, value_network):
    #     for target_param, param in zip(self.ppo.new_actor_critic.pi_parameters(), policy_network.parameters()):
    #         target_param.data.copy_(param.data)
    #     for target_param, param in zip(self.ppo.new_actor_critic.V_parameters(), value_network.parameters()):
    #         target_param.data.copy_(param.data)
    #     self.ppo.copy_target()

    def step(self, x, random=False, test=False):
        """predict action based on input and update internal storage"""
        if random:
            action = np.random.randint(0, self.output_size * self.output_size, size=(x.size(0),))
        else:
            self.states.append(x)
            logits = self.pi(x)
            self.logits.append(logits)
            action = self.policy(logits, test)
            self.actions.append(action)
            action = action.detach().cpu().numpy()
        # normalize actions to environment range
        action = np.unravel_index(action, (self.output_size, self.output_size))
        action = np.array(list(zip(*action)))
        return action

    def reward(self, r, d=None):
        self.rewards.append(r)
        if d is None:
            self.dones.append(torch.zeros((r.size(0),)).to(self.device))
        else:
            self.dones.append(d)

    def update(self, metrics, final_state, scope='', skip_train=False):
        if not skip_train:
            states = torch.cat([state.unsqueeze(dim=1) for state in self.states], dim=1)
            actions = torch.cat([action.unsqueeze(dim=1) for action in self.actions], dim=1)
            rewards = torch.cat([reward.unsqueeze(dim=1) for reward in self.rewards], dim=1)
            dones = torch.cat([done.unsqueeze(dim=1) for done in self.dones], dim=1)
            exp = (states, actions, rewards, dones)
            # loss = self.ppo.update(exp, final_state, metrics, scope)
            loss = self.a2c.update(exp, final_state, metrics, scope)
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []


class DMMRLAgent():
    def __init__(self, algorithm, policy, nb_threads, nb_rollout_steps,
                 max_env_steps, state_shape, obs_shape, nb_dmm_updates,
                 batchsize, test_freq, frame_stack, max_buffer_len,
                 replay_warmup, device, callbacks):
        self.max_buffer_len = max_buffer_len
        self.replay_warmup = replay_warmup
        self.obs_shape = obs_shape
        self.algorithm = algorithm
        self.policy = policy
        self.nb_threads = nb_threads
        self.nb_rollout_steps = nb_rollout_steps
        self.max_env_steps = max_env_steps
        self.max_train_steps = int(max_env_steps // (nb_threads * nb_rollout_steps))
        self.state_shape = state_shape
        self.nb_dmm_updates = nb_dmm_updates
        self.batchsize = batchsize
        self.test_freq = test_freq
        self.k = frame_stack
        self.device = device
        self.callbacks = callbacks

    class Clone:
        def __init__(self, t, preprocess, policy, nb_rollout, k, device, rollout, buffer=None):
            """create a new environment"""
            self.t = t
            self.nb_rollout = nb_rollout
            self.rollout = rollout
            self.preprocess = preprocess
            self.policy = policy
            self.k = k
            self.device = device
            self.buffer = buffer
            if self.buffer:
                self.buffer_len = self.buffer['obs'].size(1)

        # def preprocess(self, x):
        #     return x

        def run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            # self.env = phys_env.PhysEnv()
            self.env = env
            self.env.env = phys_env.PhysEnv()
            self.buffer_idx = 0
            done = True
            avg_ep_len = AverageMeter(reset_freq=100)
            ep_len = 0
            while True:
                startq.get()
                for step in range(self.nb_rollout):
                    if done:
                        obs = self.preprocess(self.env.reset()).to(self.device)
                        stateq = deque(
                            [torch.zeros(obs.shape).to(self.device)
                             for _ in range(self.k-1)], maxlen=self.k)
                        stateq.append(obs)
                        state = torch.cat(list(stateq), dim=0)
                        avg_ep_len.update(ep_len)
                        ep_len = 0
                    self.buffer['obs'][self.t, self.buffer_idx] = self.env.state
                    self.buffer['masks'][self.t, self.buffer_idx] = self.env.obs_mask
                    #TODO:debug
                    # self.buffer['actions'][self.t, self.buffer_idx] = self.env.action
                    action = self.policy(pi(state.unsqueeze(dim=0))).detach()
                    next_obs, r, done, _ = self.env.step(action.cpu().numpy())
                    self.rollout['states'][self.t, step] = state
                    self.rollout['actions'][self.t, step] = action
                    self.rollout['rewards'][self.t, step] = r
                    self.rollout['dones'][self.t, step] = float(done)
                    self.buffer['agent_actions'][self.t, self.buffer_idx] = action.cpu()
                    self.buffer['rewards'][self.t, self.buffer_idx] = r
                    self.buffer['dones'][self.t, self.buffer_idx] = float(done)
                    obs = self.preprocess(next_obs).to(self.device)
                    stateq.append(obs)
                    state = torch.cat(list(stateq), dim=0)
                    self.buffer_idx = (self.buffer_idx + 1) % self.buffer_len
                    ep_len += 1
                # finally add the next state into the states buffer as well to do value estimation
                self.rollout['states'][self.t, self.nb_rollout] = state
                self.buffer['obs'][self.t, self.buffer_idx] = self.env.state
                self.buffer['masks'][self.t, self.buffer_idx] = self.env.obs_mask
                # TODO:debug
                # self.buffer['actions'][self.t, self.buffer_idx] = self.env.action
                self.rollout['avg_ep_len'][self.t] = avg_ep_len.avg
                stopq.put(self.t)

        def test_run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            #self.env = phys_env.PhysEnv()
            self.env = env
            self.env.env = phys_env.PhysEnv()
            while True:
                idx = startq.get()
                obs = self.preprocess(self.env.reset())
                stateq = deque(
                    [torch.zeros(obs.shape).to(self.device)
                     for _ in range(self.k - 1)], maxlen=self.k)
                ep_reward = 0
                done = False
                while not done:
                    obs = obs.to(self.device)
                    stateq.append(obs)
                    state = torch.cat(list(stateq), dim=0)
                    action = self.policy(pi(state.unsqueeze(dim=0)), test=True)
                    next_state, r, done, _ = self.env.step(action.detach().cpu().numpy())
                    ep_reward += r
                    obs = self.preprocess(next_state)
                print("Testing reward {:.3f}".format(ep_reward))
                self.rollout['test_reward'][idx] = ep_reward
                stopq.put(1)

    def create_batch(self, buffer, samples_added, seqlen, metrics):
        """
        create a batch for training DMM
        :param buffer: multithreaded replay buffer containing glimpses, masks and dones
        :param samples_added: total samples that have been added for each thread
        :param seqlen: length of sequence of states in batch
        :return: minibatch of state and masks both of shape [nb_batch, steps, *state_shape]
        """
        nb_valid_samples = min(samples_added, self.max_buffer_len)
        end_idx = (samples_added - 1) % self.max_buffer_len
        # draw some random numbers
        idxs = np.random.randint(0, nb_valid_samples - seqlen, (self.batchsize,))
        batch_idxs = []
        last_idxs = []  # final obs, mask and glimpse action idx
        batch_threads = []
        skips = 0
        for idx in idxs :
            while True:
                if samples_added > self.max_buffer_len:
                    seq_idx = range(idx + end_idx + 1, idx + end_idx + 1 + seqlen)
                    seq_idx = [i % self.max_buffer_len for i in seq_idx]
                    last_idx = (idx + end_idx + 1 + seqlen) % self.max_buffer_len
                else:
                    seq_idx = range(idx, idx + seqlen)
                    last_idx = idx + seqlen
                # pick a thread randomly
                t = np.random.randint(0, self.nb_threads)
                if any(buffer['dones'][t, seq_idx[:-1]]):  # last state can be terminal
                    idx = np.random.randint(0, nb_valid_samples - seqlen)
                    skips+=1
                else:
                    break
            batch_idxs.append(seq_idx)
            batch_threads.append(t)
            last_idxs.append(last_idx)
        metrics['map/done_skips'].update(skips)
        batch_glimpses = [buffer['obs'][t][idx].unsqueeze(0) for i, t in enumerate(batch_threads) for idx in batch_idxs[i]]
        batch_glimpses = torch.cat(batch_glimpses, dim=0)
        batch_glimpses = batch_glimpses.view(self.batchsize, seqlen, *batch_glimpses.size()[1:]).transpose(0,1)
        final_glimpses = [buffer['obs'][t][last_idxs[i]].unsqueeze(0) for i, t in enumerate(batch_threads)]
        final_glimpses = torch.cat(final_glimpses, dim=0)
        batch_masks = [buffer['masks'][t][idx].unsqueeze(0) for i, t in enumerate(batch_threads) for idx in batch_idxs[i]]
        batch_masks = torch.cat(batch_masks, dim=0)
        batch_masks = batch_masks.view(self.batchsize, seqlen, *batch_masks.size()[1:]).transpose(0,1)
        final_masks = [buffer['masks'][t][last_idxs[i]].unsqueeze(0) for i, t in enumerate(batch_threads)]
        final_masks = torch.cat(final_masks, dim=0)
        batch_actions = [buffer['actions'][t][idx].unsqueeze(0) for i, t in enumerate(batch_threads) for idx in batch_idxs[i]]
        batch_actions = torch.cat(batch_actions, dim=0)
        batch_actions = batch_actions.view(self.batchsize, seqlen).transpose(0,1)
        batch_agent_actions = [buffer['agent_actions'][t][idx].unsqueeze(0) for i, t in enumerate(batch_threads) for idx in batch_idxs[i]]
        batch_agent_actions = torch.cat(batch_agent_actions, dim=0)
        batch_agent_actions = batch_agent_actions.view(self.batchsize, seqlen).transpose(0,1)
        batch_rewards = [buffer['rewards'][t][idx].unsqueeze(0) for i, t in enumerate(batch_threads) for idx in batch_idxs[i]]
        batch_rewards = torch.cat(batch_rewards, dim=0)
        batch_rewards = batch_rewards.view(self.batchsize, seqlen).transpose(0,1)
        batch_dones = [buffer['dones'][t][idx].unsqueeze(0) for i, t in enumerate(batch_threads) for idx in batch_idxs[i]]
        batch_dones = torch.cat(batch_dones, dim=0)
        batch_dones = batch_dones.view(self.batchsize, seqlen).transpose(0,1)
        return (batch_glimpses, batch_masks, batch_actions, final_glimpses, final_masks, batch_agent_actions, batch_rewards, batch_dones)

    def train(self, make_env, preprocess):
        # create shared data between agent clones. observations are collected
        # in a replay buffer to train DMM
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
        buffer = {
            'obs': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                *self.obs_shape).to(self.device).share_memory_(),
            'masks': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                1,
                *self.obs_shape[1:]).to(self.device).share_memory_(),
            'actions': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                dtype=torch.long).to(self.device).share_memory_(),
            'agent_actions': torch.empty(
                self.nb_threads,
                self.max_buffer_len,
                dtype=torch.long).share_memory_(),
            'rewards': torch.empty(
                self.nb_threads,
                self.max_buffer_len,).to(self.device).share_memory_(),
            'dones': torch.empty(
                self.nb_threads,
                self.max_buffer_len).to(self.device).share_memory_(),
        }
        # stopqs and startqs tell when the actors should collect rollouts
        stopqs = []
        startqs = []
        # make the policy available to all processes
        self.algorithm.actor_critic.share_memory()
        procs = []
        # clones = []
        for t in range(self.nb_threads):
            startq = mp.Queue(1)
            startqs.append(startq)
            stopq = mp.Queue(1)
            stopqs.append(stopq)
            c = self.Clone(
                t, preprocess, self.policy, self.nb_rollout_steps, self.k,
                self.device, rollout, buffer)
            # clones.append(c)
            proc = mp.Process(target=c.run, args=(self.algorithm.pi, make_env(), startq, stopq))
            procs.append(proc)
            proc.start()
        # have a thread for testing
        test_startq = mp.Queue(1)
        test_stopq = mp.Queue(1)
        test_clone = self.Clone(
            t+1, preprocess, self.policy, self.nb_rollout_steps, self.k,
            self.device, rollout)
        test_proc = mp.Process(
            target=test_clone.test_run,
            args=(self.algorithm.pi, make_env(), test_startq, test_stopq))
        test_proc.start()

        # train
        step = 0
        seqlen = 10
        maxseqlen = 25
        gammas = {10: 0.983, 11: 0.965, 12: 0.952, 13: 0.941, 14: 0.933,
                  15: 0.927, 16: 0.921, 17: 0.917, 18: 0.913, 19: 0.910,
                  20: 0.908, 21: 0.906, 22: 0.904, 23: 0.902, 24: 0.901,
                  25: 0.9}
        env = make_env()
        env.map.batchsize = self.batchsize
        env.optimizer = optim.Adam(env.map.parameters(), lr=1e-6)
        env.glimpse_agent.ppo.gamma = gammas[seqlen]
        samples_added = 0
        metrics = {'agent/policy_loss': AverageMeter(),
                   'agent/val_loss': AverageMeter(),
                   'agent/policy_entropy': AverageMeter(),
                   'agent/avg_val': AverageMeter(),
                   'agent/avg_reward': AverageMeter(),
                   'agent/avg_ep_len': AverageMeter(),
                   'glimpse/policy_loss': AverageMeter(),
                   'glimpse/val_loss': AverageMeter(),
                   'glimpse/policy_entropy': AverageMeter(),
                   'glimpse/avg_val': AverageMeter(),
                   'glimpse/avg_reward': AverageMeter(),
                   'map/write cost': AverageMeter(),
                   'map/step cost': AverageMeter(),
                   'map/post write': AverageMeter(),
                   'map/post step': AverageMeter(),
                   'map/overall': AverageMeter(),
                   'map/done_skips': AverageMeter(),
                   }
        while step < self.max_train_steps:
            # start collecting data
            for start in startqs:
                start.put(1)
            # wait for the agents to finish getting data
            for stop in stopqs:
                stop.get()
            # for c in clones:
            #     c.run(self.al
            # update PPO!
            self.algorithm.update((
                rollout['states'][:,:-1],
                rollout['actions'],
                rollout['rewards'],
                rollout['dones']), rollout['states'][:,-1], metrics, scope='agent')
            metrics['agent/avg_ep_len'].update(rollout['avg_ep_len'].mean().item())
            step += 1
            samples_added += self.nb_rollout_steps
            # update DMM!
            if samples_added > self.replay_warmup:
                if rollout['avg_ep_len'].mean().item() > 2 * seqlen:
                    seqlen = min(seqlen + 1, maxseqlen)
                    env.glimpse_agent.ppo.gamma = gammas[seqlen]
                    if seqlen < maxseqlen:
                        print("Increasing sequence length to {}".format(seqlen))
                for _ in range(self.nb_dmm_updates):
                    exp = self.create_batch(buffer, samples_added, seqlen, metrics)
                    env.train_map(exp, metrics)
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


###################################################


class ReinforcePolicyDiscrete(object):
    """basic reinforce algorithm"""

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
        self.ep_rewards = []
        self.ep_log_probs = []
        self.gamma = 0.9
        self.eps = 1e-6
        self.attn_trans_matrix = np.array([[0, -10],
                                           [0,  10],
                                           [-10, 0],
                                           [10,  0]])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-6)

    def step(self, x, loc, test=False):
        """predict action based on input and update internal storage"""
        probs = self.policy(x)
        m = Categorical(probs)
        action = m.sample()
        if not test:
            self.ep_log_probs.append(m.log_prob(action))
        action = action.detach().cpu().numpy()
        action_one_hot = np.zeros((x.size(0), 4))
        action_one_hot[range(x.size(0)), action] = 1
        attn_trans = np.matmul(action_one_hot, self.attn_trans_matrix)
        loc += attn_trans.astype(np.int64)
        return loc

    def update(self):
        """called at the end of the episode to update policy"""
        R = 0
        policy_loss = []
        returns = []
        for r in self.ep_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + self.eps)
        for log_prob, R in zip(self.ep_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.ep_rewards = []
        self.ep_log_probs = []
        return policy_loss.item()

class ReinforcePolicyContinuous(object):
    """basic reinforce algorithm"""

    def __init__(self, input_size, attn_size, policy, device):
        self.input_size = float(input_size)
        self.attn_size = float(attn_size)
        self.policy = policy
        self.device = device
        self.ep_rewards = []
        self.ep_log_probs = []
        self.gamma = 0.9
        self.eps = 1e-6
        self.steps = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-6)

    def step(self, x, test=False):
        """predict action based on input and update internal storage"""
        probs = self.policy(x)
        m = MultivariateNormal(probs, 0.1*torch.eye(probs.size(1)).to(self.device))
        action = m.sample()
        one = torch.ones_like(action).to(self.device)
        action = torch.min(torch.max(action, -one), one)
        # with some probability, still take a random action
        randa = (torch.rand_like(action)*2 - 1).to(self.device)
        switch = torch.bernoulli(self.sigma * torch.ones_like(action)).to(self.device)
        action = switch * randa + (1 - switch) * action

        if not test:
            self.ep_log_probs.append(m.log_prob(action))
        # normalize to possible attention range
        # action *= ((self.input_size - self.attn_size + 1)/self.input_size)
        # normalize to image scale
        action = (action + 1)/2
        action = self.input_size * action.detach().cpu().numpy()
        return action

    @property
    def sigma(self):
        if self.steps < 200000:
            return 1 - self.steps*(1-0.05)/200000
        else:
            return 0.05

    def reward(self, r):
        self.ep_rewards.append(r)

    def update(self):
        """called at the end of the episode to update policy"""
        R = 0
        policy_loss = []
        returns = []
        for r in self.ep_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + self.eps)
        for log_prob, R in zip(self.ep_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.ep_rewards = []
        self.ep_log_probs = []
        self.steps += 1
        return policy_loss


class ReinforcePolicyRandom(object):
    """random glimpse locations"""

    def __init__(self, policy, device):
        pass

    def step(self, x, loc, test=False):
        action = np.random.rand(x.size(0), 2)
        # normalize to environment action range
        action = action * (44/64.) + (10/64.)
        action *= 64
        return action

    def update(self):
        return 0

    def reward(self, r):
        pass


