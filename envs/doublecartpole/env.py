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
from envs.env import Environment
from utils import parse_conf
# Physics constants
PI = np.pi

N_TAU=10
DELTA_T=0.02

class Doublecartpole(Environment):
    def __init__(self, rng, g,m_cart,m_pole_1,m_pole_2,l1,l2,d1,d2,d3,f,r,s):
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
        self.g = float(g)  
        self.m_cart = float(m_cart)
        self.m_pole_1 = float(m_pole_1)
        self.m_pole_2 = float(m_pole_2)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.d3 = float(d3)
        self.f = float(f)
        self.r = float(r)
        self.s = float(s)
            
           
            

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
        force = self.f
        if (action == 0):
            force = -self.f

        # Divide DELTA_T into smaller tau's, to better take into account
        # the transitions
        n_tau = N_TAU
        delta_t = DELTA_T

        tau = delta_t / n_tau
        for i in range(n_tau):
            # Physics -> See wiki for the formulas
        
            x, x_dot, theta_1, theta_1_dot,theta_2, theta_2_dot = self._last_observation#_ = self._last_observation
            cos_theta_1 = np.cos(theta_1)
            cos_theta_2 = np.cos(theta_2)
            sin_theta_1 = np.sin(theta_1)
            sin_theta_2 = np.sin(theta_2)
            cos_diff_theta = np.cos(theta_1 - theta_2)
            sin_diff_theta = np.cos(theta_1 - theta_2)
            m1 = self.m_pole_1
            m2 = self.m_pole_2
            m = self.m_cart
            l1 = self.l1
            l2 = self.l2
            d1 = self.d1
            d2 = self.d2
            d3 = self.d3
            lhs_mat = np.array([[m1 + m2 + m, l1 * (m1 + m2) * cos_theta_1, m2 * l2 * cos_theta_2],
                                [l1 * (m1 + m2) * cos_theta_1, l1 * l1 * (m1 + m2), l1 * l2 * m2 * cos_diff_theta],
                                [l2 * m2*cos_theta_2, l1 * l2 * m2*cos_diff_theta, l2 * l2 * m2]])
                                
                                
            rhs_mat = np.array([[l1 * (m1 + m2) * theta_1_dot * theta_1_dot * sin_theta_1 + m2 * l2 * theta_2_dot * theta_2_dot * sin_theta_2 - x_dot * d1 + force],
                                [-l1 * l2 * m2 * theta_2_dot * theta_2_dot * sin_diff_theta + g * (m1+m2)*l1*sin_theta_1 - d2 * theta_1_dot],
                                [l1 * l2 * m2 * theta_1_dot * theta_1_dot * sin_diff_theta + g * l2 * m2 * sin_theta_2 - d3 * theta_2]])
            sols = np.linalg.solve(lhs_mat,rhs_mat)
            x_dd = sols[0]
            theta_1_dd = sols[1]
            theta_2_dd = sols[2]    

            # Update observation vector
            self._last_observation = [
                x + tau*x_dot,
                x_dot + tau*x_dd,
                self._to_range(theta_1 + tau*theta_1_dot),
                theta_1_dot + tau*theta_1_dd,
                self._to_range(theta_2 + tau*theta_2_dot),
                theta_2_dot + tau*theta_2_dd,
                ]
    
        # Simple reward
        reward = - abs(theta_1)
        reward -= abs(theta_2) 
        reward -= abs(self._last_observation[0])/2.
        # The cart cannot move beyond -S or S
        S = float(self.s)
        if(self._last_observation[0]<-S):
            self._last_observation[0]=-S
        if(self._last_observation[0]>S):
            self._last_observation[0]=S
 
        return reward

    def convert_repr(self):
        lastobs = copy.deepcopy(self._last_observation)
    

        return lastobs
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
            mode - Not used in this example.
        """
        # Reset initial observation to a random x and theta
        if mode == -1:
            #Learning set 
            x = self._rng.uniform(-0.25, 0.25)
            theta = self._rng.uniform(-PI/4, PI/4)
        else:
            if mode == 1:
                #Validation set
                ranges_x = [(-0.5,-0.25),(0.25,0.5)]
                ranges_theta = [(-PI/2,-PI/4),(PI/4,PI/2)]
            else:
                #Test set
                ranges_x = [(-1,-0.5),(0.5,1)]
                ranges_theta = [(-PI,-PI/2),(PI/2,PI)]
            range_x = ranges_x[self._rng.randint(len(ranges_x))]
            range_theta = ranges_theta[self._rng.randint(len(ranges_theta))]
            
            x = self._rng.uniform(*range_x)
            theta = self._rng.uniform(*range_theta)
            
        self._last_observation = [x, 0, theta, 0]
        return self._last_observation
        
    def summarizePerformance(self, test_data_set, path_dump=None, prefix_file=""):
        """ This function is called at every PERIOD_BTW_SUMMARY_PERFS.

        Arguments:
            test_data_set - Simulation data returned by the agent.
        """
        Environment.summarizePerformance(self,test_data_set, path_dump, prefix_file)
        print ("Summary Perf")

        # Save the data in the correct input format for video generation
        observations = test_data_set.observations()
        data = np.zeros((len(observations[0]), len(observations)))
        for i in range(1, 4):
            data[:,i] = observations[i - 1]
        data[:,0]=np.arange(len(observations[0]))*0.02
        save_mp4(data, self._video, self.path_dump + "/" + prefix_file)
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