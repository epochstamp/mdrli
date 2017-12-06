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
import sys
sys.path.insert(0, 'utils/cartpole')
sys.path.insert(0, 'utils')
from render_movie import save_mp4
from deer.base_classes import Environment
from deer.agent import DataSet
from utils import parse_conf
# Physics constants
PI = np.pi

class Cartpole(Environment):
    def __init__(self, rng="", params=None):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        # Defining the type of environment
        self._rng = rng
        # Observations = (x, x_dot, theta, theta_dot, timestamp)
        #self._last_observation = [0, 0, 0, 0]
        self._input_dim = [(1,), (1,), (1,), (1,)]
        self._video = 0
        self.params = dict()
        self.params["G"] = 9.8 
        self.params["M_CART"] = 1.0
        self.params["M_POLE"] = 0.1
        self.params["L"] = 0.5
        self.params["F"] = 10
        self.params["R"] = 0
        self.params["S"] = 5
        self.params["VIDEO_PREFIX"] = ""
        if params is not None:
            for k,v in params.items():
                self.params[k] = float(v) if k != "VIDEO_PREFIX" else v
            
           
            

    def act(self, action):
        """ This is the most important function in the environment. 
        We simulate one time step in the environment. Given an input 
        action, compute the next state of the system (position, speed, 
        angle, angular speed) and return a reward. 
        
        Argument:
            action - 0: move left (F = -10N); 1: move right (F = +10N)
        Return:
            reward - reward for this transition
        """
        # Direction of the force
        force = self.params["F"]
        if (action == 0):
            force = -self.params["F"]

        # Divide DELTA_T into smaller tau's, to better take into account
        # the transitions
        n_tau = 100
        delta_t = 0.01
        tau = delta_t / n_tau
        for i in range(n_tau):
            # Physics -> See wiki for the formulas
        
            x, x_dot, theta, theta_dot, = self._last_observation#_ = self._last_observation
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        
        lhs_mat = np.array([[self.params["M_POLE"] + self.params["M_CART"], self.params["M_POLE"]*self.params["L"] * cos_theta],[cos_theta,-self.params["L"]]])
        rhs_mat = np.array([force - self.params["M_POLE"]*self.params["L"]*theta_dot*theta_dot*sin_theta, - self.params["G"]*sin_theta])
        sols = np.linalg.solve(lhs_mat,rhs_mat)
        x_dd = sols[0]
        theta_dd = sols[1]
        """
        f_cart = self.params["MU_C"] * np.sign(x_dot)
        f_pole = self.params["MU_P"] * theta_dot / (self.params["M_POLE"]*self.params["L"])

        tmp = (force + self.params["M_POLE"]*self.params["L"]*sin_theta*theta_dot**2 - f_cart) \
              / (self.params["M_POLE"] + self.params["M_CART"])
        theta_dd = (self.params["G"]*sin_theta - cos_theta*tmp - f_pole) \
                   / (self.params["L"]*(4/3. - self.params["M_POLE"]*cos_theta**2/(self.params["M_POLE"] + self.params["M_CART"]))) 
        x_dd = tmp - self.params["M_POLE"]*theta_dd*cos_theta/(self.params["M_POLE"] + self.params["M_CART"])
        """

        # Update observation vector
        self._last_observation = [
            x + tau*x_dot,
            x_dot + tau*x_dd,
            self._to_range(theta + tau*theta_dot),
            theta_dot + tau*theta_dd,
            ]
    
        # Simple reward
        reward = - abs(theta) 
        reward -= abs(self._last_observation[0])/2.
     
        # The cart cannot move beyond -5 or 5
        S = float(self.params["S"])
        if(self._last_observation[0]<-S):
            self._last_observation[0]=-S
        if(self._last_observation[0]>S):
            self._last_observation[0]=S
 
        return reward

    def convert_repr(self):
        lastobs = copy.deepcopy(self._last_observation)
    
        if int(self.params["R"]) >= 1:
            S = float(self.params["S"])
            lastobs[0] = S + lastobs[0]

        if int(self.params["R"]) >= 2:
            lastobs[1] *= 0.0174533 

        return lastobs
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
            mode - Not used in this example.
        """
        # Reset initial observation to a random x and theta
        x = self._rng.uniform(-1, 1)
        theta = self._rng.uniform(-PI, PI)
        self._last_observation = [x, 0, theta, 0]
        return self._last_observation
        
    def summarizePerformance(self, test_data_set):
        """ This function is called at every PERIOD_BTW_SUMMARY_PERFS.

        Arguments:
            test_data_set - Simulation data returned by the agent.
        """
        print ("Summary Perf")

        # Save the data in the correct input format for video generation
        observations = test_data_set.observations()
        data = np.zeros((len(observations[0]), len(observations)))
        for i in range(1, 4):
            data[:,i] = observations[i - 1]
        data[:,0]=np.arange(len(observations[0]))*0.02
        save_mp4(data, self._video, self.params["VIDEO_PREFIX"])
        self._video += 1
        return

    def _to_range(self, angle):
        # Convert theta in the range [-PI, PI]
        n = abs(angle) // (2*PI)
        if (angle < 0):
            angle += n*2*PI
        else:
            angle -= n*2*PI

        if (angle < -PI):
            angle = 2*PI - abs(angle)
        elif (angle > PI):
            angle = -(2*PI - angle)

        return angle

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        # The environment allows two different actions to be taken
        # at each time step
        return 2             

    def observe(self):
        return self.convert_repr() 

if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    env = MyEnv(rng)
    env.reset()
    dataset = DataSet(env)
    
    
    for i in range(1000):
        act = 0
        r = env.act(act)
        obs = env.observe()
        dataset.addSample(obs, act, r, False, 0)
    env.summarizePerformance(dataset)
