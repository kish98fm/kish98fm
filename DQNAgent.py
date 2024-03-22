import tensorflow as tf
import numpy as np

import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense
from collections import deque

import random

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = Dense(64, activation='relu')
        self.layer2 = Dense(64, activation='relu')
        self.value = Dense(2)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value

class Agent:
    def __init__(self):
        # hyper parameters
        self.lr =0.001
        self.gamma = 0.99

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = optimizers.adam_v2.Adam(learning_rate=self.lr)

        self.batch_size = 64
        self.state_size = 4
        self.action_size = 2
        self.episodes = 0

        self.memory = deque(maxlen=2000)

    def next_episode(self):
        self.episodes += 1

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        q_value = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return int(action)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        self.dqn_target.save(name,save_format="tf")

    def load(self, name):
        self.dqn_target = load_model(name)
        self.dqn_model = load_model(name)
        
    def update(self):
        if len(self.memory) > self.batch_size:
            mini_batch = random.sample(self.memory, self.batch_size)

            states = [i[0] for i in mini_batch]
            actions = [i[1] for i in mini_batch]
            rewards = [i[2] for i in mini_batch]
            next_states = [i[3] for i in mini_batch]
            dones = [i[4] for i in mini_batch]

            dqn_variable = self.dqn_model.trainable_variables
            with tf.GradientTape() as tape:
                tape.watch(dqn_variable)

                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                dones = tf.convert_to_tensor(dones, dtype=tf.float32)

                # Calculate the target Q-values for the next states
                target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))
                next_action = tf.argmax(target_q, axis=1)
                target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)
                target_value = (1-dones) * self.gamma * target_value + rewards

                # Calculate Q values for current state
                main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
                main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * main_q, axis=1)

                # Loss function
                error = tf.square(main_value - target_value) * 0.5
                error = tf.reduce_mean(error)
                
            dqn_grads = tape.gradient(error, dqn_variable)
            self.opt.apply_gradients(zip(dqn_grads, dqn_variable))
            if self.episodes % 20 == 0:
                self.update_target()