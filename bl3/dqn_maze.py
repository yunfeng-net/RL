import gymnasium as gym

from dqn import DQNAgent
from two_dim import MazeEnv

env = MazeEnv(1)
model = DQNAgent(env, lr=0.01, buffer_type=2,steps=3)
model.learn(reward_th=9)
#model.save("dqn_maze")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_maze")

obs, info = env.reset()
env.render()
while True:
    action, _states = model.predict(obs)
    #action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.render()
        obs, info = env.reset()
        env.render()
