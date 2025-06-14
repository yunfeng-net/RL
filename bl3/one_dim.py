import gymnasium as gym
import random
 

# Define your problem using python and Farama-Foundation's gymnasium API:
#定义环境
class SimpleCorridor(gym.Env):
    def __init__(self, len:int=28):
        # 初始化环境，包括设置结束位置、当前位置、动作空间（两个离散动作：左和右）和观察空间。
        self.end_pos = len
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # left and right
        self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,))

    def reset(self, *, seed=None, options=None):
        # 重置环境，将当前位置设为0，并返回初始观察值。
        """Resets the episode.
        Returns:
           Initial observation of the new episode and an info dict.
        """
        self.cur_pos = random.randint(0, self.end_pos)
        # Return initial observation.
        return [self.cur_pos], {} # obs, info

    def step(self, action):
        # 根据给定的动作在环境中执行一步操作。根据动作和当前位置更新智能体位置。
        # 当到达走廊末端（目标）时，设置terminated标志。
        # 当目标达成时，奖励为+1.0，否则为-0.1。
        # 返回新的观察值、奖励、terminated标志、truncated标志和信息字典。

        """Takes a single step in the episode given `action`.
        Returns:
            New observation, reward, terminated-flag, truncated-flag, info-dict (empty).
        """
        # Walk left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Walk right.
        elif action == 1:
            self.cur_pos += 1
        # Set `terminated` flag when end of corridor (goal) reached.
        terminated = self.cur_pos >= self.end_pos
        truncated = False
        # +1 when goal reached, otherwise -1.
        reward = 1.0 if terminated else -1e-2
        #reward = 1.0 if action==1 else -0.1
        return [self.cur_pos], reward, terminated, truncated, {}
    
    def render(self):
        print(self.cur_pos)
