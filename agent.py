from __future__ import print_function
import numpy as np
import time
import sys

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((env.state_num, 4))
        self.EPSILON = 0.1
        self.ALPHA = 0.1
        self.GAMMA = 0.9

    def policy(self, observation):
        state = self.env.state_id(observation)
        if (np.random.uniform() > 1 - self.EPSILON) or ((self.Q[state, :] == 0).all()):
            action = np.random.randint(0, 4)  # 0~3
        else:
            action = self.Q[state, :].argmax()
        return action

    def update(self, observation, action, observation2, reward, done):
        state = self.env.state_id(observation)
        new_state = self.env.state_id(observation2)
        f = self.Q[state, action]
        self.Q[state, action] = (1 - self.ALPHA) * self.Q[state, action] + \
            self.ALPHA * (reward + self.GAMMA * self.Q[new_state, :].max())
        #print("change q[",state,"=>",new_state, "] from ", f, "=>", self.Q[state, action])

    def save(self, name):
        np.savetxt(name, self.Q)
        return
    def load(self, name):
        self.Q = np.loadtxt(name)
        return

def qlearning(agent, env, num_episodes, max_number_of_steps):
    for episode in range(num_episodes):
        observation = env.reset()
        #print(observation)
        ok = 0
        for i in range(max_number_of_steps):
            #env.render()
            action = agent.policy(observation)
            observation2, reward,done, info = env.step(action)
            agent.update(observation, action, observation2, reward, done)
            observation = observation2
            if done:
                print('%d Episode finished after %d steps' % (episode, i + 1))
                ok = 1
                break
        if ok==0:
            print('%d Episode failed' %(episode))
def sarsa(agent, env, num_episodes, max_number_of_steps):
    for episode in range(num_episodes):
        observation = env.reset()
        #print(observation)
        ok = 0
        action = agent.policy(observation)
        for i in range(max_number_of_steps):
            #env.render()
            observation2, reward,done, info = env.step(action)
            action = agent.policy(observation)
            agent.update(observation, action, observation2, reward, done)
            observation = observation2
            if done:
                print('%d Episode finished after %d steps' % (episode, i + 1))
                ok = 1
                break
        if ok==0:
            print('%d Episode failed' %(episode))

