import gymnasium as gym

from stable_baselines3 import DQN
from two_dim import MazeEnv

env = MazeEnv()
model = DQN("MlpPolicy", env, learning_rate=0.01, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
#model.save("dqn_maze")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_maze")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    #action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()