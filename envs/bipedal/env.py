import copy
import gym
import numpy as np
from envs.env import Environment

"""
        Bipedal walker

        Authors : Samy Aittahar, Benoit Umé

        BipedalWalker proxy class environment from OpenAI Gym. See https://gym.openai.com/envs/BipedalWalker-v2/ for a brief overview.
"""


class Bipedal(Environment):
    def __init__(self, rng):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        self.env = gym.make('BipedalWalker-v2')
        self.rng=rng
        self._last_observation = self.env.reset()
        self.is_terminal=False
        self._input_dim = [(24,)]      
    def act(self, action):
        """ Simulate one time step in the environment.
        action : (act1, act2, act3, act4)
        where:
        act1 = Hip motor1 torque
        act2 = Knee motor1 torque
        act3 = Hip motor2 torque
        act4 = Knee motor2 torque
        """
        reward=0
        self._last_observation, r, self.is_terminal, info = self.env.step(action)
        reward+=r
        if(self.is_terminal==True):
            return

                
        return reward
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
        Mode : int
            -1 corresponds to training and 0 to test
        """
        self.mode=mode
        
        self._last_observation = self.env.reset()
        if (self.mode==-1): # Reset to a random value when in training mode (that allows to increase exploration)
            high=self.env.observation_space.high
            low=self.env.observation_space.low
            self._last_observation=low+self.rng.rand(2)*(high-low)            
            self.env.state=self._last_observation

        self.is_terminal=False
        return self._last_observation
                
    def inTerminalState(self):
        return self.is_terminal

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        """
        You can act on 4 motors
        """
        return 4  

    def observe(self):
        """
        Observation contains 24 elements
        """
        return copy.deepcopy(self._last_observation)
        
    def summarizePerformance(self, test_data_set,path_dump=None):
        Environment.summarizePerformance(self,test_data_set,path_dump)
        states = test_data_set.states()
        actions = test_data_set.actions()
        rewards = test_data_set.rewards()
        terminals = test_data_set.terminals()
        #XXX - You are supposed to plot somewhere some graphs relevant 
        #to this environment
        #It can be a video, or a human-friendly graph... Here you go !
        
if __name__=="__main__":
        env = Bipedal(np.random.RandomState())
        env.reset()
        print (env.observe().shape)
        
