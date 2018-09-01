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

class GAgent(object):
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.01
        self.batch_size = 5
        self.model = self.build_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def build_model(self):
        H = 10
        model = Sequential()
        model.add(Dense(H, input_dim=self.state_size, activation='relu', kernel_initializer="glorot_normal"))
        model.add(Dense(1, activation='sigmoid', kernel_initializer="glorot_normal"))
        model.compile(loss='binary_crossentropy', #loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def reshape(self, observation):
        return np.reshape(observation, [1, self.state_size])
    def policy(self, observation):
        state = self.reshape(observation)
        tfprob = self.model.predict(state)
        return 1 if np.random.uniform() < tfprob else 0
    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr //= np.std(discounted_epr)        
        return discounted_r

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
                t = self.model.predict(self.reshape(observation2))[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        #print("change q[",state,"=>",new_state, "] from ", f, "=>", self.Q[state, :])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




