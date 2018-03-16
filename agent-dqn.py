# -*- coding: utf-8 -*-
from rle_python_interface import RLEInterface
import sys
import gc
import random
from random import randrange
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint

EPISODES = 500000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        #self.checkpointer = ModelCheckpoint(filepath='weights-f_zero.hdf5', verbose=1, save_best_only=True)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return randrange(self.action_size)

        act_values = self.model.predict(state)
        print('predicting...')
        #print('act_values[0].shape', act_values[0].shape)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action_idx, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            #print('reward', reward)
            #print('predict.shape', self.model.predict(next_state).shape)
            #print('target_f.shape', target_f.shape)
            #print('target', target)
            #print('action', action_idx)
            target_f[0, action_idx] = target
            self.model.fit(state, target_f, epochs=1, verbose=1)
        self.save('f_zero.hdf5')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'rom_file', 'core_file')
        sys.exit()

    rle = RLEInterface()
    rle.setInt(b'random_seed', 42)
    USE_SDL = True
    if USE_SDL:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            #rle.setBool(b'sound', False) # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            #rle.setBool(b'sound', True)
            rle.setBool(b'display_screen', True)

    # Load the ROM file
    rle.loadROM(sys.argv[1], sys.argv[2])

    width, height = rle.getScreenDims()
    print("width", width)
    print("height", height)
    state_size = (height, width, 4)
    normalized_size = 255/2

    #state_size = env.observation_space.shape[0]
    action_size = rle.getLegalActionSet().shape[0]
    print('action_size', action_size)
    #action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    def get_next_state(rle):
        state = rle.getScreenRGB() / normalized_size - 1
        return np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])

    for e in range(EPISODES):
        rle.reset_game()
        state = get_next_state(rle)
        done = False
        total_reward = 0
        while not done:
            action_idx = agent.act(state)
            action = rle.getLegalActionSet()[action_idx]
            reward = rle.act(action)
            if reward != 0:
              print('reward', reward)
            reward = reward if not done else -10
            total_reward += reward
            next_state = get_next_state(rle)
            done = rle.game_over()
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state

        print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, total_reward, agent.epsilon))
        print('Garbage collection:', gc.collect())
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
