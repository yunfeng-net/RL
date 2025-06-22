import numpy as np
import random
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffer import ReplayBuffer, PrioritizedReplayBuffer

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 超参数
BUFFER_SIZE = int(1e6)  # 回放缓冲区大小
BATCH_SIZE = 32        # 小批量大小
GAMMA = 0.99            # 折扣因子
TAU = 1e-3              # 软更新参数
#TAU = 5e-2              # 软更新参数
LR_ACTOR = 1e-4         # 演员网络学习率
LR_CRITIC = 1e-3        # 评论家网络学习率
WEIGHT_DECAY = 0        # L2权重衰减


# 演员网络 (策略网络)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# 评论家网络 (Q网络)
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Ornstein-Uhlenbeck 噪声过程
class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

# DDPG 智能体
class DDPGAgent:
    def __init__(self, vec_env, lr=1e-3, gamma=0.99, buffer=0):
        self.state_size = vec_env.observation_space.shape[0]
        self.action_size = vec_env.action_space.shape[0]
        self.lr = lr
        self.gamma = gamma
        self.env = vec_env
        self.policy_update_freq = 2
        self.device = torch.device("cpu")
        
        # 演员网络 (策略网络)
        self.actor_local = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # 评论家网络 (Q网络)
        self.critic_local = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # 噪声过程
        self.noise = OUNoise(self.action_size, 0)
        
        # 回放缓冲区
        if buffer==0:
            self.memory = ReplayBuffer(BUFFER_SIZE)
        else:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE)
    
    
    def predict(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1), state
    
    def reset(self):
        self.noise.reset()
    
    def update(self, ith):
        # 如果缓冲区中有足够的经验，则学习
        if len(self.memory) <= BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        gamma = self.gamma
        # 更新评论家网络
        # 获取下一个状态的动作和Q值
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next).squeeze(1)
        # 计算Q目标值
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # 计算当前Q值
        Q_expected = self.critic_local(states, actions).squeeze(1)
        # 计算损失
        critic_loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
        # 计算 TD-error 并更新优先级
        td_errors = (Q_targets - Q_expected).abs().detach().numpy().flatten()
        self.memory.update_priorities(idxs, td_errors)

        # 最小化损失
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新演员网络
        if ith % self.policy_update_freq!=0:
            return
        
        # 计算策略梯度
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # 最小化损失
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新目标网络 (软更新)
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # 训练循环示例
    def learn(self, episodes=1000, max_t=BATCH_SIZE):
        env = self.env
        scores = []
        scores_window = deque(maxlen=100)  # 最近100个episode的分数
        
        for i_episode in range(episodes):
            state, _ = env.reset()
            self.reset()
            score = 0
            for t in range(max_t):
                action, _ = self.predict(state)
                next_state, reward, done, _, info = env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                score += reward
                
                #if i_episode % 10 ==0:
                self.update(t)

                if done:
                    break
            

            scores_window.append(score)
            scores.append(score)
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
            if np.mean(scores_window) >= 30.0:  # 环境特定的成功标准
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')
                break
            if i_episode%1000==0:
                self.save("ddpg.pth")
        return scores

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "actor_dict": self.actor_local.state_dict(),
            "critic_dict": self.critic_local.state_dict(),
        }, path)
    
    def load(self, path: str) -> "DDPGAgent":
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint["actor_dict"])
        self.critic_local.load_state_dict(checkpoint["critic_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_dict"])
        