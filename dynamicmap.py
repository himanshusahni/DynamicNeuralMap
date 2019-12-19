import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from pytorch_rl import policies
from pytorch_rl.utils import ImgToTensor

from networks import *
from utils import MSEMasked, AverageMeter


class DynamicMap():
    def __init__(self, size, channels, env_size, env_channels, batchsize, device):
        self.size = size
        self.channels = channels
        self.env_size = env_size
        self.batchsize = batchsize
        self.device = device
        if env_size == 10 and size == 10:
            self.write_model = MapWrite_x_x(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_x_x(in_channels=channels, out_channels=env_channels)
        elif env_size == 84 and size == 21:
            self.write_model = MapWrite_84_21(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_21_84(in_channels=channels, out_channels=env_channels)
        # elif env_size == 84 and size == 84:
        #     self.write_model = MapWrite_84_84(in_channels=env_channels, out_channels=channels)
        #     self.reconstruction_model = MapReconstruction_84_84(in_channels=channels, out_channels=env_channels)
        self.step_model = MapStepResidual(in_channels=channels, out_channels=channels//2)

    def to(self, device):
        self.write_model.to(device)
        self.step_model.to(device)
        self.reconstruction_model.to(device)

    def reset(self):
        """
        reset the map to beginning of episode
        """
        self.map = torch.zeros((self.batchsize, self.channels, self.size, self.size)).to(self.device)
        # self.allmaps = [self.map,]

    def detach(self):
        """
        detaches underlying map, stopping gradient flow
        """
        self.map = self.map.detach()

    def MaskObs2Map(self, mask, minus_mask):
        '''
        converts an observation mask to one of size of map
        '''
        if self.size == self.env_size:
            minus_map_mask = minus_mask
            map_mask = mask
        elif self.size == self.env_size // 2:
            minus_map_mask = nn.MaxPool2d(2, stride=2)(minus_mask)
            map_mask = 1 - minus_map_mask
        elif self.size == self.env_size // 4:
            minus_map_mask = nn.MaxPool2d(2, stride=2)(minus_mask)
            minus_map_mask = nn.MaxPool2d(2, stride=2)(minus_map_mask)
            map_mask = 1 - minus_map_mask
        map_mask = map_mask.detach()
        minus_map_mask = minus_map_mask.detach()
        return map_mask, minus_map_mask

    def write(self, glimpse, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        w = self.write_model(glimpse)
        # write
        map_mask, minus_map_mask = self.MaskObs2Map(obs_mask, minus_obs_mask)
        w *= map_mask
        # map_clone = self.map.clone()
        # map_clone = map_clone * minus_map_mask + w
        # post_write_reconstruction = self.reconstruction_model(map_clone)
        self.map = self.map * minus_map_mask + w
        # returns a cost of writing
        # return w.abs().mean(), post_write_reconstruction
        return w.abs().mean()

    def step(self):
        """
        uses the model to advance the map by a step
        """
        # only dynamic part of map is stepped, the whole map is provided as input
        dynamic = self.step_model(self.map)
        new_map = self.map.clone()
        new_map[:, self.channels//2:, :, :] = self.map[:, self.channels//2:, :, :] + dynamic
        self.map = new_map
        # self.allmaps.append(self.map)
        return dynamic.abs().mean()

    def reconstruct(self):
        """
        attempt to reconstruct the entire state image using current map
        """
        return self.reconstruction_model(self.map)

    def reconstruct_all(self):
        """
        attempt to reconstruct the current state image using maps from previous time steps
        """
        reconstructions = []
        for i, m in enumerate(reversed(self.allmaps)):
            if i > 6:
                break
            for j in range(i):
                dynamic = self.step_model(m)
                new_map = m.clone()
                new_map[:, self.channels//2:, :, :] = m[:, self.channels//2:, :, :] + dynamic
                m = new_map
            reconstructions.append(self.reconstruction_model(m))
        return list(reversed(reconstructions))

    def parameters(self):
        return list(self.write_model.parameters()) +\
               list(self.step_model.parameters()) + \
               list(self.reconstruction_model.parameters())


    def save(self, path):
        torch.save({
            'write': self.write_model,
            'step': self.step_model,
            'reconstruct': self.reconstruction_model
        }, path)

    def load(self, path):
        models = torch.load(path, map_location='cpu')
        self.write_model = models['write']
        self.step_model = models['step']
        self.reconstruction_model = models['reconstruct']


class ConditionalDynamicMap(DynamicMap):
    def __init__(self, size, channels, env_size, env_channels, nb_actions, batchsize, device):
        self.size = size
        self.channels = channels
        self.env_size = env_size
        self.batchsize = batchsize
        self.device = device
        if env_size == 10 and size == 10:
            self.write_model = MapWrite_x_x(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_x_x(in_channels=channels, out_channels=env_channels)
        elif env_size == 84 and size == 21:
            self.write_model = MapWrite_84_21(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_21_84(in_channels=channels, out_channels=env_channels)
        # elif env_size == 84 and size == 84:
        #     self.write_model = MapWrite_84_84(in_channels=env_channels, out_channels=channels)
        #     self.reconstruction_model = MapReconstruction_84_84(in_channels=channels, out_channels=env_channels)
        self.step_model = MapStepResidualConditional(in_channels=channels, out_channels=channels//2, in_size=size, nb_actions=nb_actions)

    def step(self, a):
        """
        uses the model to advance the map by a step
        """
        # only dynamic part of map is stepped, the whole map is provided as input
        dynamic = self.step_model(self.map, a)
        new_map = self.map.clone()
        new_map[:, self.channels//2:, :, :] = self.map[:, self.channels//2:, :, :] + dynamic
        self.map = new_map
        # self.allmaps.append(self.map)
        return dynamic.abs().mean()


class BlendDNM(DynamicMap):

    def __init__(self, size, channels, env_size, env_channels, batchsize, device):
        super().__init__(size, channels, env_size, env_channels, batchsize, device)
        self.reconstruction_blend = MapBlendSpatial(in_channels=channels * 2, out_channels=channels)

    def to(self, device):
        super().to(device)
        self.reconstruction_blend.to(device)

    def reset(self):
        """
        reset the map to beginning of episode
        """
        super().reset()
        self.obs = torch.zeros((self.batchsize, self.channels, self.size, self.size)).to(self.device)
        self.allobs = [self.obs,]

    def write(self, glimpse, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        w = self.write_model(glimpse)
        self.obs = w.clone()
        self.allobs.append(self.obs)
        # blend together the old dynamic info and the new dynamic info
        w[:, self.channels//2:, :, :] = self.blend_model(w[:, self.channels//2:, :, :], self.map[:, self.channels//2:, :, :])
        # write
        map_mask, minus_map_mask = self.MaskObs2Map(obs_mask, minus_obs_mask)
        w *= map_mask
        self.map = self.map * minus_map_mask + w
        # returns a cost of writing
        return w.abs().mean()

    def reconstruct(self):
        """
        attempt to reconstruct the entire state image using current map
        """
        blend = self.reconstruction_blend(self.map, self.obs)
        return self.reconstruction_model(blend)

    def parameters(self):
        return super().parameters() + list(self.reconstruction_blend.parameters())

    def save(self, path):
        torch.save({
            'write': self.write_model,
            'step': self.step_model,
            'blend': self.blend_model,
            'reconstruct': self.reconstruction_model,
            'reconstruct_blend': self.reconstruction_blend
        }, path)

    def load(self, path):
        models = torch.load(path, map_location='cpu')
        self.write_model = models['write']
        self.step_model = models['step']
        self.blend_model = models['blend']
        self.reconstruction_model = models['reconstruct']
        self.reconstruction_blend = models['reconstruct_blend']


class DNMNoMask(DynamicMap):
    """Basically a DNM w/o write constraints"""
    def write(self, glimpse, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        w = self.write_model(glimpse)
        # blend together the old dynamic info and the new dynamic info
        w[:, self.channels//2:, :, :] = self.blend_model(w[:, self.channels//2:, :, :], self.map[:, self.channels//2:, :, :])
        self.map = w
        # returns a cost of writing
        return w.abs().mean()


class SpatialNet():
    """reimplementation of Spatial Net"""
    def __init__(self, size, channels, env_size, env_channels, batchsize, device):
        self.size = size
        self.channels = channels
        self.env_size = env_size
        self.batchsize = batchsize
        self.device = device
        if env_size == 10 and size == 10:
            self.write_model = MapWrite_x_x(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_x_x(in_channels=channels, out_channels=env_channels)
        elif env_size == 84 and size == 21:
            self.write_model = MapWrite_84_21(in_channels=env_channels, out_channels=channels)
            self.reconstruction_model = MapReconstruction_21_84(in_channels=channels, out_channels=env_channels)
        # elif env_size == 84 and size == 84:
        #     self.write_model = MapWrite_84_84(in_channels=env_channels, out_channels=channels)
        #     self.reconstruction_model = MapReconstruction_84_84(in_channels=channels, out_channels=env_channels)
        self.step_model = MapStepSpatial(in_channels=channels, out_channels=channels)
        self.blend_model = MapBlendSpatial(in_channels=channels * 2, out_channels=channels)

    def to(self, device):
        self.write_model.to(device)
        self.step_model.to(device)
        self.reconstruction_model.to(device)
        self.blend_model.to(device)

    def reset(self):
        """
        reset the map to beginning of episode
        """
        self.map = torch.zeros((self.batchsize, self.channels, self.size, self.size)).to(self.device)
        self.allmaps = [self.map,]
        self.obs = torch.zeros((self.batchsize, self.channels, self.size, self.size)).to(self.device)
        self.allobs = [self.obs,]

    def write(self, glimpse, obs_mask, minus_obs_mask):
        """
        stores an incoming glimpse into the memory map
        :param glimpse: a (batchsize, *attention_dims) input to be stored in memory
        :param attn: indices of above glimpse in map coordinates (where to write)
        """
        # what to write
        self.obs = self.write_model(glimpse)
        self.allobs.append(self.obs)
        return 0

    def step(self):
        """
        uses the model to advance the map by a step
        """
        self.map = self.step_model(self.map, self.obs)
        self.allmaps.append(self.map)
        return self.map.abs().mean()

    def reconstruct(self):
        """
        attempt to reconstruct the entire state image using current map
        """
        blend = self.blend_model(self.map, self.obs)
        return self.reconstruction_model(blend)

    def parameters(self):
        return list(self.write_model.parameters()) + \
               list(self.step_model.parameters()) + \
               list(self.blend_model.parameters()) + \
               list(self.reconstruction_model.parameters())

    def save(self, path):
        torch.save({
            'write': self.write_model,
            'step': self.step_model,
            'blend': self.blend_model,
            'reconstruct': self.reconstruction_model
        }, path)

    def load(self, path):
        models = torch.load(path, map_location='cpu')
        self.write_model = models['write']
        self.step_model = models['step']
        self.blend_model = models['blend']
        self.reconstruction_model = models['reconstruct']


class MapEnvironment():
    """
    creates an environment style interface of getting
    map states from dyanmic map of underlying attention masked
    environment.
    """
    def __init__(self, map, glimpse_agent, env_size, attn_size, device):
        self.map = map
        self.env_size = env_size
        self.attn_size = attn_size
        self.glimpse_agent = glimpse_agent
        self.glimpse_policy = policies.MultinomialPolicy()
        self.device = device

    def preprocess(self, img):
        img = ImgToTensor()(img)
        return img/127.5 - 1

    def glimpse_step(self):
        #TODO: debug
        # self.logits = self.glimpse_agent.pi(self.map.map.detach())
        # self.action = self.glimpse_policy(self.logits).detach()
        # action = self.action.cpu().numpy()
        # # normalize actions to environment range
        # loc = np.unravel_index(action, (self.env_size, self.env_size))
        agent_loc = (self.env.player_body.position.x, 84-self.env.player_body.position.y)
        goal_loc = (self.env.target_body.position.x, 84-self.env.target_body.position.y)
        if self.ep_step == 0 or self.ep_step == 2:
            # look at agent
            loc = agent_loc
        elif self.ep_step == 1 or self.ep_step == 3:
            # look at goal
            loc = goal_loc
        else:
            if np.random.random() < 0.4:
                # look at agent
                loc = agent_loc
            else:
                p = np.random.random()
                if p < 0.5:
                    # pick a random location between agent and goal
                    c = np.random.random()
                    locx = agent_loc[0] + c * (goal_loc[0] - agent_loc[0])
                    locy = agent_loc[1] + c * (goal_loc[1] - agent_loc[1])
                    loc = (locx, locy)
                elif 0.5 < p < 0.7:
                    # look at goal again
                    loc = goal_loc
                else:
                    # look around randomly
                    locx = 84 * np.random.random()
                    locy = 84 * np.random.random()
                    loc = (locx, locy)
            # add a little bit of noise here
            locx = loc[0] + 10 * np.random.randn()
            locy = loc[1] + 10 * np.random.randn()
            loc = (locx, locy)
        return np.clip(loc, self.attn_size//2, self.env_size - 1 - self.attn_size//2).astype(np.int64)  # clip to avoid edges

    def glimpse_write(self, state):
        # pick next location to glimpse
        self.loc = self.glimpse_step()
        # mask it
        self.obs_mask = torch.zeros(1, self.env_size, self.env_size)
        self.obs_mask[:,
        self.loc[0] - self.attn_size//2 : self.loc[0] + self.attn_size//2 + 1,
        self.loc[1] - self.attn_size//2 : self.loc[1] + self.attn_size//2 + 1] = 1
        self.obs_mask = self.obs_mask.to(self.device)
        minus_obs_mask = 1 - self.obs_mask
        glimpse = state * self.obs_mask
        # now write the glimpse to the map
        self.map.write(glimpse.unsqueeze(dim=0), self.obs_mask, minus_obs_mask)
        return glimpse

    def reset(self):
        self.ep_step = 0
        # first reset the underlying environment and get a state
        state = self.env.reset()
        self.state = self.preprocess(state).to(self.device)
        # write to map!
        self.map.reset()
        self.glimpse_write(self.state)
        # and return the underlying map
        return self.map.map.detach().squeeze()

    def step(self, action, detach=True):
        self.ep_step += 1
        # first advance the map a step forward
        # self.map.step()
        onehot_action = torch.zeros((1,4)).to(self.device)
        onehot_action[0, action] = 1
        self.map.step(onehot_action)
        if detach:
            # no need to store gradient information for rollouts
            self.map.detach()
        # now step in the environment and get next observation as usual
        state, r, done, _ = self.env.step(action)
        self.state = self.preprocess(state).to(self.device)
        # write to map!
        self.glimpse_write(self.state)
        # and return the underlying state
        return self.map.map.detach().squeeze(), r, done, _

    def train_map(self, exp, metrics):
        state_batch, batch_masks, batch_actions, final_states, final_masks, batch_agent_actions, batch_rewards, batch_dones = exp
        mse = MSEMasked()
        mse_unmasked = nn.MSELoss()
        total_write_loss = 0
        total_step_loss = 0
        total_post_write_loss = 0
        total_post_step_loss = 0
        overall_reconstruction_loss = 0
        loss = 0
        seq_len = state_batch.size(0)
        # initialize map with initial input glimpses
        self.map.reset()
        self.optimizer.zero_grad()
        # get an empty reconstruction
        post_step_reconstruction = self.map.reconstruct()
        for t in range(seq_len):
            obs_mask = batch_masks[t]
            minus_obs_mask = 1 - obs_mask
            post_step_loss = mse(post_step_reconstruction, state_batch[t], obs_mask)
            overall_reconstruction_loss += mse_unmasked(state_batch[t], post_step_reconstruction).item()
            # save the loss as a reward for glimpse agent
            self.glimpse_agent.states.append(self.map.map.detach())
            self.glimpse_agent.reward(post_step_loss.detach() + batch_rewards[t], batch_dones[t])
            post_step_loss = post_step_loss.mean()
            # write new observation to map
            obs = state_batch[t] * obs_mask
            write_cost = self.map.write(obs, obs_mask, minus_obs_mask)
            post_write_reconstruction = self.map.reconstruct()
            post_write_loss = mse(post_write_reconstruction, state_batch[t], obs_mask).mean()
            # step forward the internal map
            # step_cost = self.map.step()
            onehot_action = torch.zeros(batch_agent_actions.size(1), 4).to(self.device)
            onehot_action[range(onehot_action.size(0)), batch_agent_actions[t]] = 1
            step_cost = self.map.step(onehot_action)
            post_step_reconstruction = self.map.reconstruct()
            # add up all losses
            loss += 0.01 * (write_cost + step_cost) + post_write_loss + post_step_loss
            total_write_loss += 0.01 * write_cost.item()
            total_step_loss += 0.01 * + step_cost.item()
            total_post_write_loss += post_write_loss.item()
            total_post_step_loss += post_step_loss.item()
        # finally post step reconstruction for final states must be done
        obs_mask = final_masks
        post_step_loss = mse(post_step_reconstruction, final_states, obs_mask).mean()
        loss += post_step_loss
        overall_reconstruction_loss += mse_unmasked(final_states, post_step_reconstruction).item()
        total_post_step_loss += post_step_loss.item()
        # finally update
        loss.backward()
        self.optimizer.step()
        # finally update the glimpse agent as well
        self.glimpse_agent.actions = batch_actions.transpose(0,1)
        self.glimpse_agent.update(metrics, self.map.map.detach(), skip_train=True, scope='glimpse')
        # glimpse agent does not get updated with the final observation because the terminal effect
        # of a glimpse is felt with one step delay. So, the final state glimpse agent
        # made a decision on was at map(seqlen-1)
        # (map(seqlen-1) -> action(seqlen) -> obs(seqlen) -> agent_action(seqlen) -> reward, terminal (seqlen + 1).
        # Hence the map(seqlen) is passed as a final state to the update method of the glimpse agent, even though
        # we know what action the agent took on it and what curiosity reward it receives (because we know obs(seqlen + 1))
        # we do not know if seqlen + 1 terminal is true or if seqlen was terminal then there's no need to update beyond it.

        metrics['map/write cost'].update(total_write_loss/seq_len)
        metrics['map/step cost'].update(total_step_loss/seq_len)
        metrics['map/post write'].update(total_post_write_loss/seq_len)
        metrics['map/post step'].update(total_post_step_loss/seq_len)
        metrics['map/overall'].update(overall_reconstruction_loss/seq_len)


class AttentionEnvironment(MapEnvironment):
    """
    Creates an attention masked environment that returns the observations
    from the environments masked by attention located at where the map agent predicts
    """

    def reset(self):
        # first reset the underlying environment and get a state
        state = self.env.reset()
        state = self.preprocess(state).to(self.device)
        # write to map!
        self.map.reset()
        glimpse = self.glimpse_write(state)
        return (glimpse + 1) * 127.5

    def step(self, action, detach=True):
        # first advance the map a step forward
        self.map.step()
        if detach:
            # no need to store gradient information for rollouts
            self.map.detach()
        # now step in the environment and get next observation as usual
        state, r, done, _ = self.env.step(action)
        state = self.preprocess(state).to(self.device)
        # write to map!
        glimpse = self.glimpse_write(state)
        return (glimpse + 1) * 127.5, r, done, _
