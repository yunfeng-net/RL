from __future__ import print_function
import numpy as np
import time
import sys

class Agent(object):
    def __init__(self, env):
        self.env = env
        #self.Q = np.zeros((env.state_num, 4))
        self.Q = np.random.uniform(low=-1, high=1, size=(env.state_num, env.action_space.n))
        self.EPSILON = 0.1
        self.ALPHA = 0.1
        self.GAMMA = 0.9

    def policy(self, observation):
        state = self.env.state_id(observation)
        if (np.random.uniform() > 1 - self.EPSILON) or ((self.Q[state, :] == 0).all()):
            action = np.random.randint(0, self.env.action_space.n)
            #print("random %d" % action)
        else:
            action = self.Q[state, :].argmax()
        return action

    def update(self, observation, action, observation2, action2, reward, done):
        state = self.env.state_id(observation)
        new_state = self.env.state_id(observation2)
        f = self.Q[state, action]
        if action2>=0:
            next_f = self.Q[new_state, action2]
        else:
            next_f = self.Q[new_state, :].max()
        self.Q[state, action] = (1 - self.ALPHA) * self.Q[state, action] + \
            self.ALPHA * (reward + self.GAMMA * next_f)
        #print("change q[",state,"=>",new_state, "] from ", f, "=>", self.Q[state, :])

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
            agent.update(observation, action, observation2, -1, reward, done)
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
            action2 = agent.policy(observation2)
            agent.update(observation, action, observation2, action2, reward, done)
            observation = observation2
            action = action2
            if done:
                print('%d Episode finished after %d steps' % (episode, i + 1))
                ok = 1
                break
        if ok==0:
            print('%d Episode failed' %(episode))


