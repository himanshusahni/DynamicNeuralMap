import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

import torch
import torch.nn.functional as F

from pytorch_rl.utils import ImgToTensor
from pytorch_rl.policies import MultinomialPolicy
from phys_env.phys_env import PhysEnv

from dynamicmap import DynamicMap
import utils
from rl import GlimpseAgent, AttentionConstrainedEnvironment

# torch.manual_seed(123)
# np.random.seed(123)

d = '/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_DMM_refactored2/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
attn_size = 21
size = 84
map = DynamicMap(
    size=21,
    channels=48,
    env_size=84,
    env_channels=3,
    batchsize=1,
    nb_actions=4,
    device=device)

mse = torch.nn.MSELoss(reduction='none')
mse_masked = utils.MSEMasked()
for step in range(12600, 12700, 100):
    ac = torch.load(os.path.join(d, 'actor_critic_{}.pth'.format(step)), map_location=device)

    model_dir = d
    path = os.path.join(model_dir, 'map_{}.pth'.format(step))
    map.load(path)
    map.to(device)
    path = os.path.join(model_dir, 'glimpse_{}.pth'.format(step))
    glimpsenet = torch.load(path, map_location='cpu')
    glimpse_pi = glimpsenet['policy_network'].to(device)
    glimpse_V = glimpsenet['value_network'].to(device)
    # glimpse_pi = glimpsenet.policy_head
    # glimpse_V = glimpsenet.value_head
    glimpse_agent = GlimpseAgent(
        output_size=84,
        attn_size=attn_size,
        batchsize=1,
        policy_network=glimpse_pi,
        value_network=glimpse_V,
        device=device)
    # glimpse_agent.load(glimpse_pi, glimpse_V)
    policy = MultinomialPolicy()
    env = AttentionConstrainedEnvironment(env_size=84, attn_size=21, device=device)
    env.env = PhysEnv()

    steps = 64
    fig, axarr = plt.subplots(steps, 4, figsize=(2*3, 2*steps))

    map.reset()
    # starting glimpse location
    glimpse_logits = glimpse_agent.pi(map.map.detach())
    glimpse_action = glimpse_agent.policy(glimpse_logits, test=False).detach()
    glimpse_action_clipped = glimpse_agent.norm_and_clip(glimpse_action.cpu().numpy())
    print(glimpse_action_clipped)
    obs, unmasked_obs, mask = env.reset(loc=glimpse_action_clipped)
    done = False
    overall_error = 0
    total_post_step_loss = 0
    for i in range(steps):
        # display reconstruction of what map sees
        post_step_reconstruction = map.reconstruct().detach().squeeze()
        overall_error += mse(post_step_reconstruction, unmasked_obs).mean().item()
        total_post_step_loss += mse_masked(post_step_reconstruction, obs, mask).mean().item()
        # first the display heatmap used for attention
        heatmap = F.softmax(glimpse_logits, dim=-1).view(size, size).detach().cpu().numpy()
        axarr[i, 0].imshow(heatmap)
        # display full environment image
        axarr[i, 1].imshow(env.env.render())
        # mark attention on environment image
        loc = glimpse_action_clipped
        attention = patches.Rectangle(
            ((loc[1]-attn_size//2), (loc[0]-attn_size//2)),
            attn_size, attn_size, linewidth=2, edgecolor='y', facecolor='none')
        axarr[i, 1].add_patch(attention)
        # mask just to make sure it matches
        axarr[i, 2].imshow(mask.cpu().numpy().squeeze())
        # write observation to map
        map.write(obs.unsqueeze(dim=0), mask, 1 - mask)
        axarr[i, 3].imshow(utils.postprocess(map.reconstruct().detach().cpu().numpy()).squeeze())
        # take a step in the environment!
        state = map.map.detach()
        logits = ac.pi(state)
        print(policy.probs(logits))
        print(ac.V(state))
        action = policy(logits)
        print(action.detach().cpu().numpy())
        # step the map forward according to agent action
        onehot_action = torch.zeros((1, 4)).to(device)
        onehot_action[0, action] = 1
        map.step(onehot_action)
        # no need to store gradient information for rollouts
        map.detach()
        # glimpse agent decides where to look after map has stepped
        glimpse_logits = glimpse_agent.pi(map.map.detach())
        glimpse_action = glimpse_agent.policy(glimpse_logits, test=False).detach()
        print(glimpse_agent.policy.entropy(glimpse_logits))
        print(glimpse_logits)
        glimpse_action_clipped = glimpse_agent.norm_and_clip(glimpse_action.cpu().numpy())
        (next_obs, next_unmasked_obs, next_mask), r, done, _ = env.step(action.cpu().numpy(), loc=glimpse_action_clipped)
        print(r)
        obs = next_obs
        mask = next_mask
        unmasked_obs = next_unmasked_obs
        if done:
            break
    print(i)
    print(overall_error/i)
    print(total_post_step_loss/i)
    plt.savefig(os.path.join(model_dir, 'rl_visualization_{}.jpeg'.format(step)))
