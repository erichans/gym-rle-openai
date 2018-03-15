#https://gym.openai.com/docs/

import gym
import gym_rle


#env = gym.make('Aladdin-v0')
#[print(env) for env in gym.envs.registry.all()]
env = gym.make('StreetFighterIi-v0')
#env = gym.make('CartPole-v0')
#env = gym.make('Wolfenstein-ram-v0')
episode_count = 100

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

agent = Agent(env.action_space)
episode_count = 10

for episode in range(episode_count):
    ob = env.reset()
    reward = 0
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(ob, reward, done)
        observation, reward, done, _ = env. step(action)
        total_reward += reward
    print('Episode', episode, 'ended with score:', total_reward)
env.close()
