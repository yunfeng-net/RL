import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from typing import Optional, Tuple, Dict, Any, Union
import gymnasium as gym
from collections import deque
import warnings
from typing import Callable
import torch.nn.functional as F
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import gym
def entropy(prob_dist):
    # 避免log(0)导致NaN，使用微小偏移量eps
    eps = 1e-10
    return -torch.sum(prob_dist * torch.log(prob_dist + eps), dim=-1)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor网络 (策略)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic网络 (状态值函数)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, state):
        features = self.shared_layers(state)
        
        # 策略输出 (未归一化的log概率)
        policy_logits = self.actor(features)
        
        # 状态值估计
        state_value = self.critic(features)
        
        return policy_logits, state_value

class A2CAgent:
    def __init__(self, vec_env, gamma=0.99, lr=0.001):
        self.env = vec_env
        self.state_dim = vec_env.observation_space.shape[0]
        self.action_dim = vec_env.action_space.n
        self.gamma = gamma  # 折扣因子
        self.lr = lr        # 学习率
        
        # 初始化网络
        self.model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def predict(self, state, epsilon : float =0, action = None, deterministic=False):
        # 使用epsilon-greedy策略
        if random.random() < epsilon:
            if action is not None:
                probs = np.full((self.env.action_space.n),0.5/self.env.action_space.n, dtype=np.float64)
                probs[action] += 0.5
                return self.env.action_space.sample(probability=probs), state
            else:
                return self.env.action_space.sample(), state

        state = torch.FloatTensor(state).unsqueeze(0)
        policy_logits, _ = self.model(state)
        
        # 使用epsilon-greedy策略
        #if np.random.rand() < epsilon:
        #    return np.random.randint(self.action_dim), state
        
        # 从策略中采样动作
        policy = F.softmax(policy_logits, dim=-1)
        if deterministic:
            policy.argmax().item(), state
        else:
            action = torch.multinomial(policy, 1).item()
        return action, state
    
    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # 反向计算优势函数
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                last_advantage = delta + self.gamma * last_advantage
            advantages[t] = last_advantage
        
        return advantages
    
    def train(self):
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(np.array(self.dones))
        
        # 计算当前状态值和下一状态值
        _, current_values = self.model(states)
        _, next_values = self.model(next_states)
        
        # 计算优势函数
        #if torch.equal(states,next_states):
        #    advantages = rewards.numpy()
        #else:
        advantages = self.compute_advantages(
            rewards.numpy(), 
            current_values.detach().numpy(),
            next_values.detach().numpy(),
            dones.numpy()
        )
        advantages = torch.FloatTensor(advantages)
        
        # 计算策略损失 (Actor)
        policy_logits, _ = self.model(states)
        policy = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        #selected_log_probs = advantages * log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        selected_log_probs = advantages * policy.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -selected_log_probs.mean()

        en = entropy(policy)
        entropy_loss = -en.mean()
        
        # 计算值函数损失 (Critic)
        a = current_values.squeeze(1)
        b = rewards + self.gamma * next_values.squeeze(1) * (1 - dones)
        value_loss = F.mse_loss(a, b.detach())
        #print(a[0],b[0],advantages,value_loss)
        #print(policy, advantages)
        # 总损失
        #if torch.equal(states,next_states):
        #    total_loss = policy_loss #+ 0.2*entropy_loss
        #else:
        total_loss = policy_loss + value_loss #+ 0.2*entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def learn(self, num_episodes=1000, max_steps=50):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        state, _ = self.env.reset()
        action = None
        for episode in range(num_episodes):
            episode_reward = 0
            #self.env.render()
            
            for step in range(max_steps):
                # 选择动作
                action, _ = self.predict(state) #, epsilon, action)
                
                # 执行动作
                next_state, reward, done, _, _ = self.env.step(action)
                #self.env.render()
                
                # 存储转移
                self.store_transition(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    action = None
                    break
            
            # 训练模型
            policy_loss, value_loss, entropy_loss = self.train()
            print(f"Episode: {episode}, Reward: {episode_reward}, "
                f"Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}, Epsilon: {epsilon:.2f}, Entropy Loss: {entropy_loss:.3f}")
                # 衰减探索率
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if done:
                state, _ = self.env.reset()
        
