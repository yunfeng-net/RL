from stable_baselines3 import PPO
import gymnasium as gym
from ddpg import DDPGAgent
env = gym.make("Pendulum-v1") #, render_mode="human")
#env = gym.make("BipedalWalker-v3") #, render_mode="human")
model = DDPGAgent(env, buffer=1, lr=1e-4)
#model.load("dppg.pth")
model.learn(episodes=5000)
model.save("dppg.pth")

# 测试模型
env = gym.make("Pendulum-v1", render_mode="human")
#env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, add_noise=False)
    obs, _, done, _, _ = env.step(action)
    #env.render()
    if done: 
        obs, _ = env.reset()