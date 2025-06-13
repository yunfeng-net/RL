import gymnasium as gym
from a2c import A2C
from one_dim import SimpleCorridor
from two_dim import MazeEnv
from a2c_simple import A2CAgent
# 创建环境
env = gym.make("CartPole-v1") #, render_mode="human")
#env = SimpleCorridor()
#env = MazeEnv(1)
model = A2CAgent(env, lr=0.01)

# 训练模型
model.learn(num_episodes=10_000, max_steps=50)

# 保存模型
#model.save("a2c_cartpole")

# 加载模型
#model = A2C.load("a2c_cartpole", env)

# 评估模型
#env = SimpleCorridor()
env = gym.make("CartPole-v1", render_mode="human")
#env = MazeEnv(0)
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
        env.render()

env.close()