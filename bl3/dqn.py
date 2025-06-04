import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
from buffer import ReplayBuffer, PrioritizedReplayBuffer
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


def weighted_average(model, weight, reference):
    """
    内存友好的实现，不创建完整的新模型副本
    :param reference_model: 用作参考结构的模型
    """
    averaged_params = {}
    
    # 对每个参数进行加权平均
    for name, param in reference.named_parameters():
        # 初始化为零
        averaged_param = torch.zeros_like(param.data)
        # 加权求和
       
        averaged_param = weight * dict(model.named_parameters())[name].data + (1.0-weight)*param.data
        averaged_params[name] = averaged_param
    
    # 加载到参考模型
    model.load_state_dict(averaged_params)
    return model
# DQN算法实现
class DQNAgent:
    def __init__(self, vec_env, lr=1e-3, gamma=0.99, 
                 batch_size=64, buffer_capacity=10000, target_update=100, buffer_type=1):
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

        if buffer_type==1:
            self.memory = ReplayBuffer(buffer_capacity)
        else:
            self.memory = PrioritizedReplayBuffer(buffer_capacity)

        
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
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            #q_values = self.target_net(next_states)
            #next_q, _ = q_values.max(dim=1, keepdim=True) # target network
            act = self.policy_net(next_states).argmax(dim=1,keepdim=True)
            q_values = self.target_net(next_states)
            next_q = q_values.gather(1,act) # double Q
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算 TD-error 并更新优先级
        td_errors = (target_q - current_q).abs().detach().numpy().flatten()
        self.memory.update_priorities(idxs, td_errors)

        # 计算损失
        #loss = F.mse_loss(current_q, target_q)
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            weighted_average(self.target_net, 0.5, self.policy_net)
            #self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def learn(self, reward_th=5):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        episodes = 5000
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
            if avg_reward >= reward_th and epsilon<0.1:
                print(f"Solved at episode {episode}!")
                #torch.save(agent.policy_net.state_dict(), 'dqn_cartpole.pth')
                break
