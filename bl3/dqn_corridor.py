import gymnasium as gym

from dqn import DQNAgent
from one_dim import SimpleCorridor

#env = gym.make("CartPole-v1", render_mode="human")
env = SimpleCorridor()
model = DQNAgent(env, lr=0.01)
model.learn()

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()