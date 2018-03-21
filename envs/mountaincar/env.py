""" The environment simulates the behavior of an inverted pendulum.
The goal of the agent, as suggested by the reward function, is 
to balance a pole on a cart that can either move left or right.

Code is based on the following inverted pendulum implementations
in C : http://webdocs.cs.ualberta.ca/%7Esutton/book/code/pole.c
in Python : https://github.com/toddsifleet/inverted_pendulum

Please refer to the wiki for a complete decription of the problem.

Author: Aaron Zixiao Qiu
(Slight) modification of the dynamics : Samy Aittahar 
"""



import numpy as np
import copy
import math
from envs.env import Environment
from gym import spaces
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'utils/mountaincar')
sys.path.insert(0, 'utils')
from render_movie import save_mp4
class Mountaincar(Environment):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, rng,min_p=-1.2,max_p=0.6,max_s=0.07,goal=0.5,rand_min_p=-0.6,rand_max_p=-0.4,rand_min_p_valid=-0.6,rand_max_p_valid=-0.4,rand_min_p_test=-0.6,rand_max_p_test=-0.4,high_divisor=1,velocity_multiplier = 1):
        self.rng = rng
        self.divisor = 2
        
        self.min_position = float(min_p)
        self.max_position = float(max_p)
        self.max_speed = float(max_s)
        self.goal_position = float(goal)

        self.rand_p = dict()
        self.rand_p[-1] = (rand_min_p,rand_max_p)
        self.rand_p[1] = (rand_min_p_valid,rand_max_p_valid)
        self.rand_p[0] = (rand_min_p_test,rand_max_p_test)
        
        self._input_dim  = [(1,),(1,)]

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.divisor = float(high_divisor)
        self.velo_multi = float(velocity_multiplier)

        self.viewer = None
        self.is_terminal = False
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)
        self.reset()       
           
            

    def act(self, action):
        """ This is the most important function in the environment. 
        We simulate one time step in the environment. Given an input 
        action, compute the next state of the system (position, speed, 
        angle, angular speed) and return a reward. 
        
        Argument:
            action - 0: move left ; 1: nothing ; 2: move right
        Return:
            reward - reward for this transition
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001*self.velo_multi + math.cos(3*position)*(-0.0025/self.divisor)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        self.is_terminal = bool(position >= self.goal_position)
        reward = -1
        self.state = (position, velocity)
        return reward
    
    #np.array(self.state)

    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
        Mode : int
            - not used
        """
        low,high = self.rand_p[mode]
        self.state = np.array([self.rng.uniform(low=low, high=high), 0])
        self.is_terminal=False
        return np.array(self.state)
                
    def inTerminalState(self):
        return self.is_terminal

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        """
        You can act on 4 motors
        """
        return 3

    def observe(self):
        """
        Observation contains 24 elements
        """
        return copy.deepcopy(np.array(self.state))
    
    def _height(self, xs):
        return np.sin(3 * xs)*.45/self.divisor+.55

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos)/self.divisor)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        
    def summarizePerformance(self, test_data_set,path_dump=None, prefix_file=""):
#        Environment.summarizePerformance(self,test_data_set,path_dump)
#        observations = test_data_set.observations()
#        rewards = test_data_set.rewards()
#        plt.plot(rewards)
        
        Environment.summarizePerformance(self,test_data_set, path_dump, prefix_file)
                # Save the data in the correct input format for video generation
        observations = test_data_set.observations()
        data = np.zeros((len(observations[0]), len(observations)+1))
        for i in range(1, 3):
            data[:,i] = observations[i - 1]
        data[:,0]=np.arange(len(observations[0]))*0.02
        save_mp4(self,data, 0, self.path_dump + "/" + prefix_file)
        return
        
        #XXX - You are supposed to plot somewhere some graphs relevant 
        #to this environment
        #It can be a video, or a human-friendly graph... Here you go !
        
if __name__=="__main__":
        env = Mountaincar(np.random.RandomState())
        env.reset()
        print (env.observe().shape)
