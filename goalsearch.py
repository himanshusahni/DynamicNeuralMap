import numpy as np

import gym

class GoalSearchEnv(object):

    def __init__(self, size):
        self.size = size
        self.attention_size = 5
        self.map = np.zeros((self.size, self.size, 5))
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, (5, self.size, self.size), dtype=np.int32)

    def reset(self):
        self.ep_step = 0
        self.map = np.zeros((self.size, self.size, 5))
        wall_x = np.random.randint(1, self.size - 1)
        wall_height = np.random.randint(1, self.size - 1)
        self.map[wall_x, 0:wall_height, 4] = 1  # wall on channel 4
        self.agent_x = 0
        self.agent_y = self.size - 1
        self.map[self.agent_x, self.agent_y, 3] = 1  # agent on channel 3
        self.attention_x, self.attention_y = self.clip_attention(self.agent_x, self.agent_y)
        # pick goal type
        self.goal = np.random.randint(2)
        self.map[1, self.size-1, self.goal] = 1  # goal indicator either on 0 or 1
        # randomly select the goal locations
        self.left_goal_x = np.random.randint(wall_x)
        self.left_goal_y = np.random.randint(wall_height)
        self.map[self.left_goal_x, self.left_goal_y, 2] = 1  # left and right goal on 2
        self.right_goal_x = np.random.randint(wall_x+1, self.size)
        self.right_goal_y = np.random.randint(wall_height)
        self.map[self.right_goal_x, self.right_goal_y, 2] = 1  # left and right goal on 2
        return self.get_obs()

    def clip_attention(self, x, y):
        """make sure attention location avoids edge case"""
        return np.clip(x, self.attention_size//2, self.size - 1 - self.attention_size//2), \
               np.clip(y, self.attention_size//2, self.size - 1 - self.attention_size//2)

    def get_obs(self):
        """return observation under current attention"""
        return self.map[(self.attention_x - self.attention_size//2):(self.attention_x + 1 + self.attention_size//2),
               (self.attention_y - self.attention_size//2):(self.attention_y + 1 + self.attention_size//2), :]
        # return self.map

    def render(self, map=None):
        if map is None:
            map = self.map
            size = self.size
        else:
            size = map.shape[0]
        # first convert map into binary
        # (look at https://gist.github.com/frnsys/91a69f9f552cbeee7b565b3149f29e3e) for this magic
        map_onehot = np.zeros_like(map)
        indices = np.argmax(map, axis=-1)
        x = np.arange(size).reshape((size, 1))
        y = np.tile(np.arange(size), (size, 1))
        map_onehot[x, y, indices] = 1
        # but could have empty space! if max probability thing is < 0.5, it's empty
        map_onehot *= (map > 0.5)
        map = map_onehot
        # now create an image out of it
        map_image = 255 * np.ones((size, size, 3))
        # first find where the agent is
        map_image[np.where(map[:, :, 3])[0], np.where(map[:, :, 3])[1], :] = [255, 255, 0]  # agent is yellow
        # now draw in the goals
        map_image[np.where(map[:, :, 0])[0], np.where(map[:, :, 0])[1], :] = [255, 0, 255]  # left goal is pink
        map_image[np.where(map[:, :, 1])[0], np.where(map[:, :, 1])[1], :] = [0, 0, 255]  # right goal is blue
        map_image[np.where(map[:, :, 2])[0], np.where(map[:, :, 2])[1], :] = [0, 255, 0]  # goal is green
        # draw in the walls black
        map_image[np.where(map[:, :, 4])[0], np.where(map[:, :, 4])[1], :] = [0, 0, 0]
        # multiply the map image onto itself
        map_image = np.repeat(map_image, 40, axis=0)
        map_image = np.repeat(map_image, 40, axis=1)
        return np.rot90(map_image.astype(np.uint8), k=1)

    def get_reward_done(self):
        if (self.agent_x == self.left_goal_x) and (self.agent_y == self.left_goal_y):
            if self.goal == 0:
                return 1, True
            else:
                return -1, True
        if (self.agent_x == self.right_goal_x) and (self.agent_y == self.right_goal_y):
            if self.goal == 1:
                return 1, True
            else:
                return -1, True
        # end due to timesteps
        if self.ep_step < 200:
            return 0, False
        else:
            return 0, True

    def step(self, action):
        """agent takes an action"""
        if action == 0:  # up
            new_y = min(self.agent_y + 1, self.size - 1)
            new_x = self.agent_x
        elif action == 1:  # down
            new_y = max(self.agent_y - 1, 0)
            new_x = self.agent_x
        elif action == 2:  # left
            new_x = max(self.agent_x - 1, 0)
            new_y = self.agent_y
        elif action == 3:  # right
            new_x = min(self.agent_x + 1, self.size - 1)
            new_y = self.agent_y
        else:
            raise ValueError("action not recognized")
        # check if wall is in place
        if self.map[new_x, new_y, 4]:
            new_x, new_y = self.agent_x, self.agent_y
        # move agent to new location!
        self.map[self.agent_x, self.agent_y, 3] = 0
        self.map[new_x, new_y, 3] = 1
        self.agent_x, self.agent_y = new_x, new_y

        r, done = self.get_reward_done()
        # attention (for now) moves to a random location
        self.attention_x, self.attention_y = self.clip_attention(
            np.random.randint(self.size), np.random.randint(self.size))
        self.ep_step += 1
        return self.get_obs(), r, done, None
#TODO: test the environment with a simple DQN agent with fully observable states, make sure it can learn!
