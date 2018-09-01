from __future__ import print_function
import numpy as np
import time
import sys
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

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

class DAgent(object):
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self.build_model()
        self.target_model = self.build_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def reshape(self, observation):
        return np.reshape(observation, [1, self.state_size])
    def policy(self, observation):
        state = self.reshape(observation)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def update(self, observation, action, observation2, action2, reward, done):
        reward = reward if not done else -10
        self.memory.append((observation, action, observation2, action2, reward, done))
        if done:
            self.target_model.set_weights(self.model.get_weights())
            return
        if len(self.memory) <= self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for observation, action, observation2, action2, reward, done in minibatch:
            state = self.reshape(observation)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(self.reshape(observation2))[0]
                #print(next_state,target)
                #print(reward + self.gamma * np.amax(t))
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #print("eps:",self.epsilon)
        #print("change q[",state,"=>",new_state, "] from ", f, "=>", self.Q[state, :])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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


