import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from buffer import ReplayBuffer, PrioritizedReplayBuffer

# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

# Q Network (Critic)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda:0")
        self.iter = 0
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        return action
    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, _ = self.actor.forward(state)
        action = torch.tanh(mean)
        return action.detach().cpu().numpy()[0], None
    def update(self, batch, replay_buffer):
        state, action, reward, next_state, done, idxs, weights = batch
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next_target = self.critic1_target(next_state, next_action)
            q2_next_target = self.critic2_target(next_state, next_action)
            q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * q_next_target
        current_q1 = self.critic1(state, action)
        critic1_loss = (weights * F.mse_loss(current_q1, target_q)).mean()
        current_q2 = self.critic2(state, action)
        critic2_loss = (weights * F.mse_loss(current_q2, target_q)).mean()

        td_errors = (target_q - current_q1).abs().detach().cpu().numpy().flatten()
        replay_buffer.update_priorities(idxs, td_errors)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        new_action, log_prob = self.actor.sample(state)
        q1_new_action = self.critic1(state, new_action)
        q2_new_action = self.critic2(state, new_action)
        q_new_action = torch.min(q1_new_action, q2_new_action)
        actor_loss = (self.alpha * log_prob - q_new_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha.to(self.device) * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().to(self.device)
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def learn(self, env, max_episodes, max_steps, batch_size, buffer=1):
        if buffer == 0:
            replay_buffer = ReplayBuffer(1_000_000)
        else:
            replay_buffer = PrioritizedReplayBuffer(1_000_000)
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward
                if len(replay_buffer) > batch_size:
                    batch = replay_buffer.sample(batch_size)
                    self.update(batch, replay_buffer)
                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    print(f"Episode {episode}, Reward: {episode_reward}")
                    break
                state = next_state
        return episode_rewards

if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("BipedalWalker-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim)
    rewards = agent.learn(env, max_episodes=1500, max_steps=300, batch_size=64)
    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs, _ = env.reset()
    while True:
        action, _ = agent.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
