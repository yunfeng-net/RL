from __future__ import print_function
import copy
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time

MAP = \
    '''
.........
.       .
.     o .
.       .
.........
'''

MAP2 = \
    '''
.........
.  x    .
.   x o .
.       .
.........
'''


DX = [-1, 1, 0, 0]
DY = [0, 0, -1, 1]


class MazeEnv(gym.Env):
    def __init__(self, mm):
        m = mm.strip().split('\n')
        m = [[c for c in line] for line in m]
        self.map = copy.deepcopy(m)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([len(self.map)-2, len(self.map[0])-2])
        self.seed()
        self.state_num = (len(self.map)-2)*(len(self.map[0])-2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def state_id(self, state):
        return (len(self.map[0])-2)*state[0]+state[1]

    def reset(self):
        for _ in range(10):
            self.state = self.observation_space.sample()
            s = self.state
            #print(s,s[0]+1,s[1]+1,self.map[s[0]+1][s[1]+1])
            if self.map[s[0]+1][s[1]+1]==' ':
                return np.array(s)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        s = self.state
        new_x = s[0] + DX[action]
        new_y = s[1] + DY[action]
        new_pos_char = self.map[new_x+1][new_y+1]
        is_end = False
        if new_pos_char == '.':
            reward = 0  # do not change position
        elif new_pos_char == ' ':
            self.state = [new_x, new_y]
            reward = 0
        elif new_pos_char == 'o':
            self.state = [new_x, new_y]
            self.map[new_x][new_y] = ' '  # update map
            is_end = True  # end
            reward = 100
        elif new_pos_char == 'x':
            self.state = [new_x, new_y]
            self.map[new_x][new_y] = ' '  # update map
            reward = -50
        return np.array(self.state), reward, is_end, {}

    def render(self):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.state[0]+1][self.state[1]+1] = 'A'
        print('\n'.join([''.join([c for c in line]) for line in printed_map]))
        time.sleep(0.1)

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
class CartPoleEnv(gym.Wrapper):
    def __init__(self, env):
        super(CartPoleEnv,self).__init__(env)
        self.state_num = 4 ** 4
    def state_id(self, state):
        cart_pos, cart_v, pole_angle, pole_v = state
        digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
        return sum([x * (4 ** i) for i, x in enumerate(digitized)])
    def step(self, action):
        observation2, reward,done, info = super(CartPoleEnv,self).step(action)
        if done:
            reward = -200 # done means failure
        return observation2, reward,done, info
    
def make_maze(name):
    if name=="space0":
        return MazeEnv(MAP)
    elif name=="space1":
        return MazeEnv(MAP2)
    elif name=="space2":
        return CartPoleEnv(gym.make('CartPole-v0'))
