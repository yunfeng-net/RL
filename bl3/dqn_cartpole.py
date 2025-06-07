import gymnasium as gym

from dqn import DQNAgent
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def make_env(env_id: str, rank: int, render_mode: str = None, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "CartPole-v1" #reward_th = 100
    env_id = "Acrobot-v1"
    env_id = "MountainCar-v0"
    #num_cpu = 6  # Number of processes to use
    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = gym.make(env_id) #, render_mode="human")
    model = DQNAgent(env, lr=0.01, buffer_type=2, steps=5, dfs=1, n_steps=2000)
    model.learn(reward_th=-500)

    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()