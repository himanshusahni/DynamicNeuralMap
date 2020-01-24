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
from rl import OffPolicyGlimpseAgent, AttentionConstrainedEnvironment
from networks import PolicyFunction_21_84

# torch.manual_seed(123)
# np.random.seed(123)

d = '/home/himanshu/experiments/DynamicNeuralMap/PhysEnv/RL_DMM_refactored_offpolicyglimpse2/'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

for step in range(200, 300, 100):
    ac = torch.load(os.path.join(d, 'actor_critic_{}.pth'.format(step)), map_location=device)

    model_dir = d
    path = os.path.join(model_dir, 'map_{}.pth'.format(step))
    map.load(path)
    map.to(device)
    path = os.path.join(model_dir, 'glimpse_{}.pth'.format(step))
    glimpsenet = torch.load(path, map_location='cpu')
    glimpse_q = glimpsenet['q'].to(device)
    glimpse_agent = OffPolicyGlimpseAgent(
        output_size=84,
        attn_size=attn_size,
        batchsize=1,
        q_arch=PolicyFunction_21_84,
        channels=48,
        device=device, )
    glimpse_agent.dqn.q = glimpse_q
    glimpse_agent.pi = glimpse_q
    # glimpse_agent.load(glimpse_pi, glimpse_V)
    policy = MultinomialPolicy()
    env = AttentionConstrainedEnvironment(env_size=84, attn_size=21, device=device)
    env.env = PhysEnv()

    steps = 20
    fig, axarr = plt.subplots(steps, 4, figsize=(2*3, 2*steps))

    map.reset()
    # starting glimpse location
    glimpse_logits = glimpse_agent.pi(map.map.detach())
    glimpse_action = glimpse_agent.policy(glimpse_logits, test=False).detach()
    glimpse_action_clipped = glimpse_agent.norm_and_clip(glimpse_action.cpu().numpy())
    print(glimpse_action_clipped)
    obs, _, mask = env.reset(loc=glimpse_action_clipped)
    done = False
    overall_error = 0
    for i in range(steps):
        # first the display heatmap used for attention
        heatmap = F.softmax(glimpse_logits, dim=-1).view(size, size).detach().cpu().numpy()
        import pdb; pdb.set_trace()
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
        # take a step in the environment!
        state = map.map.detach()
        logits = ac.pi(state)
        action = policy(logits, test=True)
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
        print(glimpse_action_clipped)
        (next_obs, _, next_mask), r, done, _ = env.step(action.cpu().numpy(), loc=glimpse_action_clipped)
        # next display reconstruction of what map sees
        reconstruction = utils.postprocess(map.reconstruct().detach().cpu().numpy()).squeeze()
        overall_error += (((env.env.render() - reconstruction)/255.)**2).mean()
        axarr[i, 3].imshow(reconstruction)
        print(r)
        obs = next_obs
        mask = next_mask
        if done:
            break
    print(overall_error/i)
    plt.savefig(os.path.join(model_dir, 'rl_visualization_{}.jpeg'.format(step)))
