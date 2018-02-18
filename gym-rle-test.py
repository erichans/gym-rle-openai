import gym
import gym_rle
env = gym.make('MortalKombat-v0')
#env = gym.make('ClassicKong-v0')
env.render()

env.reset()
while True:
    action = 0
    env.step(action)
    env.render()
