from __future__ import print_function
import numpy as np
import time
import sys
import random
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K

def my_loss(y_true, y_pred):
    return -K.mean(y_true * K.log(y_pred))

class A2CAgent(object):
    def __init__(self, env, learning_rate=0.01, reward_decay=0.95):
        self.env = env
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.obs, self.pgs, self.rs, self.probs = [], [], [], []
        self.build_model()

    def build_model(self):
        self.state = Input(shape=(self.state_size,))
        layer1 = Dense(20, init='he_uniform',
            activation='relu')(self.state)
        self.action_probability = Dense(self.action_size, init='he_uniform',
            activation='softmax')(layer1)
        self.actor = Model(inputs=self.state, outputs=self.action_probability)
        self.actor.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
              #loss='categorical_crossentropy',
              loss=my_loss,
              metrics=['accuracy'])

        layer11 = Dense(20, init='he_uniform',
            activation='tanh')(self.state)
        self.v = Dense(1, init='he_uniform')(layer11)
        self.critic = Model(inputs=self.state, outputs=self.v)
        self.critic.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=['accuracy'])

    def load(self, name):
        self.actor.load_weights(name)
    def save(self, name):
        self.actor.save_weights(name)
    def policy(self, observation):
        prob = self.actor.predict(observation[np.newaxis, :])
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
        y = np.zeros([self.action_size])
        y[action] = 1
        self.pg = np.array(y).astype('float32') #- prob
        self.prob = prob
        return action

    def store_transition(self, s, a, r):
        self.obs.append(s)
        self.pgs.append(self.pg)
        self.rs.append(r)
        self.probs.append(self.prob)


    def learn(self):
        discounted_rs_norm = self._discount_and_norm_rewards()
        a = []
        f = self.critic.predict(np.vstack(self.obs))
        #print("f ",f, len(f))
        #print("r ",discounted_rs_norm, len(discounted_rs_norm))
        #print("pgs ", self.pgs, len(self.pgs))
        for i in range(len(self.pgs)):
            a.append(self.pgs[i]*(discounted_rs_norm[i]-f[i]))
        self.actor.fit(np.vstack(self.obs), np.array(a), verbose=0)
        self.critic.fit(np.vstack(self.obs), np.array(discounted_rs_norm), verbose=0)

        self.obs, self.pgs, self.rs, self.probs = [], [], [], []
        return discounted_rs_norm
    def update(self, observation, action, observation2, action2, reward, done):
        self.store_transition(observation, action, reward)
        if done:
            self.learn()
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_rs = np.zeros_like(self.rs)
        running_add = 0
        for t in reversed(range(0, len(self.rs))):
            running_add = running_add * self.gamma + self.rs[t]
            discounted_rs[t] = running_add
        # normalize episode rewards
        #discounted_rs -= np.mean(discounted_rs)
        #discounted_rs /= np.std(discounted_rs)
        return discounted_rs
