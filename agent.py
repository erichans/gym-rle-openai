#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
# Modified by: Nadav Bhonker and Shai Rozenberg
#
# This is a direct port to python of the shared library example from
# RLE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
from random import randrange
from rle_python_interface import RLEInterface

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Conv2D, Lambda, Flatten
from keras import optimizers


if len(sys.argv) < 2:
  print('Usage:', sys.argv[0], 'rom_file', 'core_file')
  sys.exit()

rle = RLEInterface()

# Get & Set the desired settings
rle.setInt(b'random_seed', 42)
#rle.setString(b'MK_player1_character', b'scorpion')

# Set USE_SDL to true to display the screen. RLE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    #rle.setBool(b'sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    rle.setBool(b'sound', True)
    rle.setBool(b'display_screen', True)

# Load the ROM file
rle.loadROM(sys.argv[1], sys.argv[2])

# Get the list of legal actions
minimal_actions = rle.getMinimalActionSet()
print("Joystick:", len(minimal_actions))

width, height = rle.getScreenDims()
print("width", width)
print("height", height)
input_shape = (height, width, 4)
model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#mean = np.mean(data['observations'], axis=0)
#std = np.std(data['observations'], axis=0) + 1e-6
#observations_dim = env.observation_space.shape[0]
state_size = rle.getScreenRGB().shape[0]
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Flatten())
#model.add(Dense(24, input_dim=self.state_size, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(clipvalue=5), metrics=['binary_accuracy'])

a = minimal_actions[randrange(len(minimal_actions))]
# Play 10 episodes
for episode in range(10):
  total_reward = 0
  reward = 0
  X = np.zeros((1, input_shape[0], input_shape[1], 4))
  while not rle.game_over():
    print('RGB:', rle.getScreenRGB().shape)
    X[0] = rle.getScreenRGB()
    model.fit(X, [reward], epochs=1)
    # Apply an action and get the resulting reward
    reward = rle.act(a)
    print('reward', reward)
    a = model.predict(X)
    total_reward += reward
    #print('Episode', episode, 'ended with score:', reward)

  print('Episode', episode, 'ended with score:', total_reward)
  rle.reset_game()
