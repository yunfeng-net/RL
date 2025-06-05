import gymnasium as gym
import random
import copy
import numpy as np

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
    def __init__(self, ty:int = 0):
        self.ty = ty
        self.action_space = gym.spaces.Discrete(4)
        self.reset_map()
        self.observation_space = gym.spaces.MultiDiscrete([len(self.map)-2, len(self.map[0])-2])
        self.state_num = (len(self.map)-2)*(len(self.map[0])-2)
    
    def reset_map(self):
        if self.ty==0:
            mm = MAP
        else:
            mm = MAP2
        m = mm.strip().split('\n')
        self.map = [[c for c in line] for line in m]
        
    def reset(self, seed: int = None, options: dict = None):
        self.reset_map()
        pos_list = []
        for i in range(self.observation_space.nvec[0]):
            for j in range(self.observation_space.nvec[1]):
                if self.map[i+1][j+1]==' ':
                    pos_list.append([i,j])
        idx = np.random.randint(0,len(pos_list)-1)
        self.state = pos_list[idx]
        return np.array(self.state), {}

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        s = self.state
        new_x = s[0] + DX[action]
        new_y = s[1] + DY[action]
        new_pos_char = self.map[new_x+1][new_y+1]
        is_end = False
        if new_pos_char == '.':
            reward = -10  # do not change position
        elif new_pos_char == ' ':
            self.state = [new_x, new_y]
            reward = -1e-3
        elif new_pos_char == 'o':
            self.state = [new_x, new_y]
            is_end = True  # end
            reward = 10
        elif new_pos_char == 'x':
            self.state = [new_x, new_y]
            reward = -50
        return np.array(self.state), reward, is_end, False, {}

    def render(self):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.state[0]+1][self.state[1]+1] = 'A'
        print('\n'.join([''.join([c for c in line]) for line in printed_map]))


