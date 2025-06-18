
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from ddpg_fol.util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results1 = np.array([]).reshape(num_episodes,0)
        self.results2 = np.array([]).reshape(num_episodes,0)

    def __call__(self, envs, policy, debug=False, visualize=False,use_step=False, save=True):

        self.is_training = False
        self.use_step = use_step
        observation = None
        result1 = []
        result2 = []

        for env in envs:
            
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done and episode_steps<40:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                observation, reward, done, info = env.step(action)
                #if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                # if info:
                #     done = True
                
                if visualize:
                    env.render(mode='human')

                # update
                if not done:
                    episode_reward += reward
                    
                episode_steps += 1

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            
            result1.append(episode_steps)
            result2.append(episode_reward)

        result1 = np.array(result1).reshape(-1,1)
        self.results1 = np.hstack([self.results1, result1])
        
        result2 = np.array(result2).reshape(-1,1)
        self.results2 = np.hstack([self.results2, result2])

        if save:
            self.save_results('{}/validate'.format(self.save_path))
        return np.mean(result1),np.mean(result2)

    def save_results(self, fn):

        y = np.mean(self.results1, axis=0)
        error=np.std(self.results1, axis=0)
                    
        x = range(0,self.results1.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average step')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'step.png')
        #savemat(fn+'step.mat', {'step':self.results1})

        y = np.mean(self.results2, axis=0)
        error=np.std(self.results2, axis=0)
                    
        x = range(0,self.results2.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'reward.png')
        #savemat(fn+'reward.mat', {'reward':self.results2})