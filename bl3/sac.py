import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

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

class CriticWithTarget:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, tau, device):
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target.load_state_dict(self.critic.state_dict())
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.tau = tau
        self.device = device
    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def optimize(self, loss, max_norm=1.0):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm)
        self.optimizer.step()

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, n_step=3):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda:0")
        self.iter = 0
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = CriticWithTarget(state_dim, action_dim, hidden_dim, lr, tau, self.device)
        self.critic2 = CriticWithTarget(state_dim, action_dim, hidden_dim, lr, tau, self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
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
    def _get_n_step_info(self, n_step_buffer, gamma):
        reward, next_state, done = n_step_buffer[-1][2], n_step_buffer[-1][3], n_step_buffer[-1][4]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        state, action = n_step_buffer[0][0], n_step_buffer[0][1]
        return state, action, reward, next_state, done
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
            q1_next_target = self.critic1.target(next_state, next_action)
            q2_next_target = self.critic2.target(next_state, next_action)
            q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * q_next_target
        current_q1 = self.critic1.critic(state, action)
        critic1_loss = (weights * F.mse_loss(current_q1, target_q)).mean()
        current_q2 = self.critic2.critic(state, action)
        critic2_loss = (weights * F.mse_loss(current_q2, target_q)).mean()
        td_errors = (target_q - current_q1).abs().detach().cpu().numpy().flatten()
        replay_buffer.update_priorities(idxs, td_errors)
        self.critic1.optimize(critic1_loss)
        self.critic2.optimize(critic2_loss)
        if self.iter % self.policy_update_freq!=0:
            return
        new_action, log_prob = self.actor.sample(state)
        q1_new_action = self.critic1.critic(state, new_action)
        q2_new_action = self.critic2.critic(state, new_action)
        q_new_action = torch.min(q1_new_action, q2_new_action)
        actor_loss = (self.alpha * log_prob - q_new_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha.to(self.device) * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().to(self.device)
        self.critic1.update_target()
        self.critic2.update_target()
    def learn(self, env, max_episodes, max_steps, batch_size, buffer=1):
        if buffer == 0:
            replay_buffer = ReplayBuffer(1_000_000)
        else:
            replay_buffer = PrioritizedReplayBuffer(1_000_000)
        episode_rewards = []
        self.iter = 0
        self.policy_update_freq = 2
        for episode in range(max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            self.n_step_buffer.clear()
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                if reward<=-100:
                    reward = -5
                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info(self.n_step_buffer, self.gamma)
                    replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
                episode_reward += reward
                self.iter += 1
                if len(replay_buffer) > batch_size:
                    batch = replay_buffer.sample(batch_size)
                    self.update(batch, replay_buffer)
                if done or step == max_steps - 1:
                    # Push remaining transitions in buffer
                    while len(self.n_step_buffer) > 0:
                        n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info(self.n_step_buffer, self.gamma)
                        replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
                        self.n_step_buffer.popleft()
                    episode_rewards.append(episode_reward)
                    print(f"Episode {episode}, Reward: {episode_reward}")
                    break
                state = next_state
        return episode_rewards

if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("BipedalWalker-v3", hardcore=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim)
    rewards = agent.learn(env, max_episodes=1500, max_steps=300, batch_size=64)
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
    obs, _ = env.reset()
    while True:
        action, _ = agent.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
