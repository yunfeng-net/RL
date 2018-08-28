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


class Env(object):
    def __init__(self, mm):
        m = mm.strip().split('\n')
        m = [[c for c in line] for line in m]
        self.map = copy.deepcopy(m)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([len(self.map)-2, len(self.map[0])-2])
        self.seed()
        self.state_num = len(self.map)-2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        for _ in range(10):
            self.state = self.observation_space.sample()
            s = self.state
            if self.map[s[0]+1]==' ' and self.map[s[1]+1]==' ':
                return np.array(s)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        s = self.state
        new_x = s[0]+1 + DX[action]
        new_y = s[1]+1 + DY[action]
        new_pos_char = self.map[new_x][new_y]
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
            reward = -5
        return np.array(self.state), reward, self.is_end, {}

    def render(self):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.state[0]+1][self.state[1]+1] = 'A'
        print('\n'.join([''.join([c for c in line]) for line in printed_map]))
        time.sleep(2)

def make_maze(name):
    if name=="space":
        return Env(MAP)
    else:
        return Env(MAP2)
