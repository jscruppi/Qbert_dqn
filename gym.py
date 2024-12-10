"""
This is the program to initially create the DQNAgent
It will also train the DQNAgent on 10000 timesteps
Saves them to an external file to be used in other calcuations

"""
import gymnasium
import ale_py
import random

#for tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.layers import Conv2D

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

#Need this to use newer versions of gym with older versions of keras and keras-rl
class GymCompatibilityWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs  # Only return observation

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done or truncated, info  # Merge done and truncated
    
    def render(self, mode='human'):
        return self.env.render()


#Makes the DQNAgent
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn

#Builds the neural network
def build_model(height, width, channels, actions):
    input_shape = (3, 210, 160, 3)
    model = Sequential()
    #add stacks layers in the neural network
    model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape))    
    model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

#this tests to make sure the ale/gym environment works
def test_env(env):

    episodes = 3
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
    
        while not done:
            env.render()
            action = random.choice([0,1,2,3,4,5])
            observation, reward, done, info = env.step(action)
            score += reward #score == score in game

        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

def main():
    #creates the game environment
    env = gymnasium.make('ALE/Qbert-v5', render_mode='rgb_array')
    env = GymCompatibilityWrapper(env)
    height, width, channels  = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n')

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])
    dqn.fit(env, nb_steps=10000, visualize=True, verbose=1) #trains the agent

    scores = dqn.test(env, nb_episodes=10, visualize=False)
    print(np.mean(scores.history['episode_reward']))

    dqn.save_weights('dqn_weights.h5f')

if __name__ == '__main__':
    main()