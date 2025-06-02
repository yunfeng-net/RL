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
# 神经网络定义
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), \
               np.array(next_state), np.array(done, dtype=np.uint8)
    
    def __len__(self):
        return len(self.buffer)

# DQN算法实现
class DQNAgent:
    def __init__(self, vec_env, lr=1e-3, gamma=0.99, 
                 batch_size=64, buffer_capacity=10000, target_update=100):
        self.env = vec_env
        self.state_dim = vec_env.observation_space.shape[0]
        self.action_dim = vec_env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        
        # 主网络和目标网络
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不更新梯度
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        
    def predict(self, state, epsilon : float =1e-3):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1), state
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.argmax().item(), state
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def learn(self):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        episodes = 500
        max_steps = 200
        
        rewards_history = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action, _ = self.predict(state, epsilon)
                
                # 执行动作
                next_state, reward, done, _, info = self.env.step(action)
                
                # 存储经验
                self.memory.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                # 更新网络
                self.update()
                
                if done:
                    break
            
            # 衰减探索率
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards_history.append(episode_reward)
            
            # 打印训练信息
            avg_reward = np.mean(rewards_history[-100:])
            print(f'Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}')
            
            # 如果问题解决则停止训练
            if avg_reward >= 5:
                print(f"Solved at episode {episode}!")
                #torch.save(agent.policy_net.state_dict(), 'dqn_cartpole.pth')
                break
