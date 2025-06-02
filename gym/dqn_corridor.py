import gymnasium as gym

from stable_baselines3 import DQN
from one_dim import SimpleCorridor

#env = gym.make("CartPole-v1", render_mode="human")
env = SimpleCorridor()
model = DQN("MlpPolicy", env, learning_rate=0.01, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_corridor")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_corridor")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()