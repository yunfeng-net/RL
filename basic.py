from __future__ import print_function
import numpy as np
import time
from maze import make_maze


EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

np.random.seed(0)

def epsilon_greedy(Q, state):
    if (np.random.uniform() > 1 - EPSILON) or ((Q[state, :] == 0).all()):
        action = np.random.randint(0, 4)  # 0~3
    else:
        action = Q[state, :].argmax()
    return action

env = make_maze("space")
Q = np.zeros((env.state_num, 4))

num_episodes = 50
max_number_of_steps = 30
for episode in range(num_episodes):
    observation = env.reset()
    for i in range(max_number_of_steps):
        env.render()
        observation, reward, done, info = env.step(action)
        if done:
            print('%d Episode finished after %f steps' % (episode, i + 1))
            break
        action = epsilon_greedy(Q, observation)
        state = observation
        new_state = observation
        Q[state, action] = (1 - ALPHA) * Q[state, action] + \
            ALPHA * (reward + GAMMA * Q[new_state, :].max())
