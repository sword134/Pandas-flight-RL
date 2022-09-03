import gym
from gym import spaces
import numpy as np
from typing import Optional
import numpy as np
from datetime import date, timedelta, time
from reward_calculation import calc_total_reward
import os
import pandas as pd

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_checker import check_env

class PTOPTEnv(gym.Env):
    def __init__(self, df):
        super(PTOPTEnv, self).__init__()
        self.render_mode = None
        self.df = df
        self.df_length = len(df.index)-1
        self.curr_progress = 0

        self.action_space = spaces.Discrete(30)
        self.observation_space  = spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.int)

    def step(self, action):

        STD = self.df.loc[self.curr_progress, "STD"]
        print(action, action+30)

        self.df.loc[self.curr_progress, "Pulltime"] = self.df.loc[self.curr_progress, "STD"]-timedelta(minutes=action+30)

        # An episode is done if the agent has reached the target
        done = True if self.curr_progress==self.df_length else False
        reward = 100000-calc_total_reward(self.df) if done else 0 
        observation = self._get_obs()
        info = {}

        self.curr_progress += 1
        print("OBS: " +str(observation))

        return observation, reward, done, info
    
    def reset(self):
        self.curr_progress = 0
        observation = self._get_obs()
        info = self._get_info()
        print("OBS: " +str(observation.shape))
        return observation

    def _get_obs(self):
        frame = np.array([
            self.df.loc[0: self.curr_progress, 'Pulltime'].values,
            self.df.loc[:, 'Pulltime'].values,
            self.df.loc[self.curr_progress: , 'Pulltime'].values,
        ], dtype='datetime64')

        obs = np.append(frame, [[self.curr_progress, 0], [0]], axis=0)
        return obs
    
    def _get_info(self):
        return {"Test": 0}


dir_path = os.path.dirname(os.path.realpath(__file__))

df_use = pd.read_csv(dir_path + "\\Flight_schedule.csv", sep=";", decimal=",")
df_use["STD"] = pd.to_datetime(df_use["STD"], format='%Y-%m-%d %H:%M:%S')
df_use["Pulltime"] = 0

env = PTOPTEnv(df=df_use)
check_env(env)

'''
episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
'''
#model = A2C('MlpPolicy', env).learn(total_timesteps=1000)