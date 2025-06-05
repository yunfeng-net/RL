import gymnasium as gym

from dqn import DQNAgent
from one_dim import SimpleCorridor

#env = gym.make("CartPole-v1", render_mode="human")
env = SimpleCorridor()
model = DQNAgent(env, lr=0.01, steps=20, buffer_type=2)
model.learn(reward_th=0.85)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()