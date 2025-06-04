import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), \
               np.array(next_state), np.array(done, dtype=np.uint8), {}, np.full(batch_size, 1.0/batch_size)
    
    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, idxs, priorities):
        pass

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # SumTree 存储，前半部是中间节点
        self.data = np.zeros(capacity, dtype=object)  # 存储数据
        self.write = 0  # 当前写入位置
        self.n_entries = 0  # 当前存储的经验数量

    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """向下检索样本，s是用于采样的随机数"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回总优先级"""
        return self.tree[0]

    def add(self, priority, data):
        """添加经验到 SumTree"""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity: # 环形覆盖
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """更新某个位置的优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """采样一个经验"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 控制优先级的权重（0~1）
        self.beta = beta  # 重要性采样权重调整
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # 初始优先级

    def push(self, state, action, reward, next_state, done):
        """存储经验，初始优先级设为最大"""
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, data)
    
    def __len__(self):
        return self.tree.n_entries
    
    def sample(self, batch_size):
        """采样一批经验，并计算重要性采样权重"""
        batch_idx = []
        priority = np.empty(batch_size, dtype=np.float32)
        batch_data = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority[i], data = self.tree.get(s)
            batch_idx.append(idx)
            batch_data.append(data)

        # 计算重要性采样权重
        probs = priority / self.tree.total()
        batch_weights = (self.tree.n_entries * probs) ** (-self.beta)
        batch_weights /= batch_weights.max()  # 归一化

        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = np.array([d[0] for d in batch_data])
        actions = np.array([d[1] for d in batch_data])
        rewards = np.array([d[2] for d in batch_data], dtype=np.float32)
        next_states = np.array([d[3] for d in batch_data])
        dones = np.array([d[4] for d in batch_data], dtype=np.uint8)

        return (states, actions, rewards, next_states, dones, batch_idx, batch_weights)

    def update_priorities(self, idxs, priorities):
        """更新采样经验的优先级（用 TD-error 更新）"""
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha)
            if priority > self.max_priority:
                self.max_priority = priority