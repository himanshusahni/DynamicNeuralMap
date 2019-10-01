import numpy as np

import gym


class DynamicObjects(object):

    def __init__(self, size):
        self.size = size
        self.map = np.zeros((self.size, self.size, 4))
        self.observation_space = gym.spaces.Box(0, 1, (self.size, self.size, 4), dtype=np.int32)
        self.r_arrow = np.ones((40,40))
        for i in range(1,20):
            self.r_arrow[20+i, 0:i-1] = 0
            self.r_arrow[20+i, 40-i+1:40] = 0
        self.l_arrow = np.ones((40,40))
        for i in range(1,20):
            self.l_arrow[i, 0:20-i+1] = 0
            self.l_arrow[i, 20+i:40] = 0
        self.u_arrow = np.ones((40,40))
        for i in range(1,20):
            self.u_arrow[0:20-i+1, i] = 0
            self.u_arrow[20+i:40, i] = 0
        self.d_arrow = np.ones((40,40))
        for i in range(1,20):
            self.d_arrow[0:i-1, 20+i] = 0
            self.d_arrow[40-i+1:40, 20+i] = 0

    def reset(self):
        self.ep_step = 0
        self.map = np.zeros((self.size, self.size, 4))
        # place border walls
        self.map[0, :, 3] = 1  # wall on channel 3
        self.map[self.size-1, :, 3] = 1  # wall on channel 3
        self.map[:, 0, 3] = 1  # wall on channel 3
        self.map[:, self.size-1, 3] = 1  # wall on channel 3
        # place vertical wall
        wall_x = np.random.randint(1, self.size - 1)
        wall_height = np.random.randint(1, self.size - 1)
        self.map[wall_x, 0:wall_height, 3] = 1  # wall on channel 3
        # place horizontal wall
        wall_y = np.random.randint(1, self.size - 1)
        wall_width = np.random.randint(1, self.size - 1)
        self.map[0:wall_width, wall_y, 3] = 1  # wall on channel 3
        # place stationary object
        o_x, o_y = np.random.randint(0, self.size, size=(2,))
        # do not spawn on top of anything else!
        while np.any(self.map[o_x, o_y]):
            o_x, o_y = np.random.randint(0, self.size, size=(2,))
        self.map[o_x, o_y, 2] = 1  # stationary objects on 2
        # spawn l-r moving objects
        self.lr_x, self.lr_y = np.random.randint(0, self.size, size=(2,))
        # do not spawn on top of anything else!
        while np.any(self.map[self.lr_x, self.lr_y]):
            self.lr_x, self.lr_y = np.random.randint(0, self.size, size=(2,))
        self.lr_vel = [np.random.choice([-1,1]), 0]
        self.map[self.lr_x, self.lr_y, 1] = self.lr_vel[0]  # lr object on 1
        # spawn u-d moving objects
        self.ud_x, self.ud_y = np.random.randint(0, self.size, size=(2,))
        # do not spawn on top of anything else!
        while np.any(self.map[self.ud_x, self.ud_y]):
            self.ud_x, self.ud_y = np.random.randint(0, self.size, size=(2,))
        self.ud_vel = [0, np.random.choice([-1,1])]
        self.map[self.ud_x, self.ud_y, 0] = self.ud_vel[1]  # ud object on 0
        return self.get_obs()

    def get_obs(self):
        """return observation under current attention"""
        return self.map

    def render(self, map=None):
        if map is None:
            map = self.map
            size = self.size
        else:
            size = map.shape[0]
        # first convert map into binary
        # (look at https://gist.github.com/frnsys/91a69f9f552cbeee7b565b3149f29e3e) for this magic
        map_onehot = np.zeros_like(map)
        indices = np.argmax(np.abs(map), axis=-1)
        x = np.arange(size).reshape((size, 1))
        y = np.tile(np.arange(size), (size, 1))
        map_onehot[x, y, indices] = 1
        # but could have empty space! if max probability thing is < 0, it's empty
        map_onehot *= (np.abs(map) > 0.5)
        # now create an image out of it
        map_image = 255 * np.ones((size*40, size*40, 3))
        # first draw in the objects
        ud_inds = np.where(map_onehot[:, :, 0])
        for ind in zip(*ud_inds):
            if map[ind[0], ind[1], 0] > 0:
                dir = self.d_arrow
            else:
                dir = self.u_arrow
            map_image[ind[0]*40:(ind[0]+1)*40, ind[1]*40:(ind[1]+1)*40, 0] = 255 * (1 - dir)  # u-d object is blue
            map_image[ind[0]*40:(ind[0]+1)*40, ind[1]*40:(ind[1]+1)*40, 1] = 255 * (1 - dir)  # u-d object is blue
        lr_inds = np.where(map_onehot[:, :, 1])
        for ind in zip(*lr_inds):
            if map[ind[0], ind[1], 1] > 0:
                dir = self.r_arrow
            else:
                dir = self.l_arrow
            map_image[ind[0]*40:(ind[0]+1)*40, ind[1]*40:(ind[1]+1)*40, 1] = 255 * (1 - dir)  # l-r object is red
            map_image[ind[0]*40:(ind[0]+1)*40, ind[1]*40:(ind[1]+1)*40, 2] = 255 * (1 - dir)  # l-r object is red
        s_inds = np.where(map_onehot[:, :, 2])
        for ind in zip(*s_inds):
            map_image[ind[0]*40:(ind[0]+1)*40, ind[1]*40:(ind[1]+1)*40, :] = [0, 255, 0]  # stationary object is green
        # then draw in the walls
        w_inds = np.where(map_onehot[:, :, 3])
        for ind in zip(*w_inds):
            map_image[ind[0]*40:(ind[0]+1)*40, ind[1]*40:(ind[1]+1)*40, :] = [0, 0, 0]  # walls are black
        # multiply the map image onto itself
        return np.rot90(map_image.astype(np.uint8))

    def get_reward_done(self):
        # end due to timesteps
        if self.ep_step < 200:
            return 0, False
        else:
            return 0, True

    def step(self):
        # now move the objects. change directions if hits the border
        if self.lr_x == self.size - 1:
            self.lr_vel[0] = -1
        elif self.lr_x == 0:
            self.lr_vel[0] = 1
        new_lr_x = self.lr_x + self.lr_vel[0]
        new_lr_y = self.lr_y + self.lr_vel[1]
        # change directions if it hits another object
        if np.any(self.map[new_lr_x, new_lr_y]):
            self.lr_vel[0] *= -1
            new_lr_x = self.lr_x + self.lr_vel[0]
            new_lr_y = self.lr_y + self.lr_vel[1]
            # but now check one last time again if this led the agent into a wall!
            if new_lr_x > (self.size - 1) or new_lr_x < 0:
                new_lr_x = self.lr_x
        self.map[self.lr_x, self.lr_y, 1] = 0
        self.lr_x = new_lr_x
        self.lr_y = new_lr_y
        self.map[self.lr_x, self.lr_y, 1] = self.lr_vel[0]
        if self.ud_y == self.size - 1:
            self.ud_vel[1] = -1
        elif self.ud_y == 0:
            self.ud_vel[1] = 1
        new_ud_x = self.ud_x + self.ud_vel[0]
        new_ud_y = self.ud_y + self.ud_vel[1]
        # change directions if it hits another object
        if np.any(self.map[new_ud_x, new_ud_y]):
            self.ud_vel[1] *= -1
            new_ud_x = self.ud_x + self.ud_vel[0]
            new_ud_y = self.ud_y + self.ud_vel[1]
            # but now check one last time again if this led the agent into a wall!
            if new_ud_y > (self.size - 1) or new_ud_y < 0:
                new_ud_y = self.ud_y
        self.map[self.ud_x, self.ud_y, 0] = 0
        self.ud_x = new_ud_x
        self.ud_y = new_ud_y
        self.map[self.ud_x, self.ud_y, 0] = self.ud_vel[1]
        r, done = self.get_reward_done()
        self.ep_step += 1
        return self.get_obs(), r, done, None


# # generate trajectories
# env = DynamicObjects(16)
# outdir = 'data-DynamicObjects-v1/'
# # import matplotlib.pyplot as plt
# import torch, os
# from pytorch_rl.utils import ImgToTensor
# preprocess = ImgToTensor()
# d = True
# ep = -1
# step = 0
# for i in range(500000):
#     if d:
#         s = env.reset()
#         d = False
#         ep += 1
#         step = 0
#         os.mkdir(os.path.join(outdir, str(ep)))
#         torch.save(torch.FloatTensor([0]*200), os.path.join(outdir, str(ep), 'actions.pt'))
#     else:
#         s, r, d, _ = env.step()
#         # plt.imshow(env.render())
#         # plt.show()
#         step += 1
#     torch.save(preprocess(s), os.path.join(outdir, str(ep), '{}.pt'.format(step)))
#     if i % 10000 == 0:
#         print(i)
