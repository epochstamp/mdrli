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
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'envs/')
sys.path.insert(0, '.')
from render_movie import save_mp4
from envs.env import Environment
from data.dataset import DataSet
# Physics constants
PI = np.pi

N_TAU=10
DELTA_T=0.02

def rescale(x,oldmax,oldmin,newmax,newmin):
    return ((newmax - newmin)/(oldmax-oldmin)) * (x-oldmax) + newmax

class Cartpole(Environment):
    def __init__(self, rng, g=9.8,m_cart=1.0,m_pole=0.1,l=0.5,min_f=-10,max_f=10,r=0,s=5,stepsize=20,number_init_states=10):
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
        self.m_pole = float(m_pole)
        self.l = float(l)
        self.r = float(r)
        self.s = float(s)
        self.min_f = min_f
        self.max_f = max_f
        self._continuous = stepsize <= 0
        self._n_actions = [[self.min_f,self.max_f]] if self._continuous else int((self.max_f - self.min_f)/stepsize) + 1
        self.stepsize = stepsize
        self.actions = None
        self.init_states = {1:None, 0:None}
        self.n_init_states = number_init_states
           
            

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
        if self._continuous:
            force = action[0]#rescale(action[0],1,-1,self.max_f,self.min_f)
        else:
            if self.actions is None:    
                
                self.actions = np.arange(self.min_f, self.max_f + self.stepsize, self.stepsize)
        # Divide DELTA_T into smaller tau's, to better take into account
        # the transitions
        n_tau = N_TAU
        delta_t = DELTA_T

        tau = delta_t / n_tau


        for i in range(n_tau):
            # Physics -> See wiki for the formulas
        
            x, x_dot, theta, theta_dot, = self._last_observation#_ = self._last_observation
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        
            lhs_mat = np.array([[self.m_pole + self.m_cart, self.m_pole*self.l * cos_theta],[cos_theta,-self.l]])
            rhs_mat = np.array([force - self.m_pole*self.l*theta_dot*theta_dot*sin_theta, - self.g*sin_theta])
            sols = np.linalg.solve(lhs_mat,rhs_mat)
            x_dd = sols[0]
            theta_dd = sols[1]
   

            # Update observation vector
            self._last_observation = [
                x + tau*x_dot,
                x_dot + tau*x_dd,
                self._to_range(theta + tau*theta_dot),
                theta_dot + tau*theta_dd,
                ]
        
        
        # The cart cannot move beyond -S or S
        S = float(self.s)
        if(self._last_observation[0]<-S):
            self._last_observation[0]=-S
        if(self._last_observation[0]>S):
            self._last_observation[0]=S


        # Simple reward
        if not self.inTerminalState():
            reward = - abs(theta) 
            reward -= abs(self._last_observation[0])/2.
        else:
            reward = -500
        

        return reward

    def convert_repr(self):
        lastobs = copy.deepcopy(self._last_observation)
    
        if int(self.r) >= 1:
            S = float(self.s)
            lastobs[0] = S + lastobs[0]

        if int(self.r) >= 2:
            lastobs[1] *= 0.0174533 

        return lastobs
                
    def reset(self, mode=-1):
        """ Reset environment for a new episode.

        Arguments:
            mode - Not used in this example.
        """
        if mode != -1:
            if self.init_states[mode] is None:
                
                self.init_states[mode] = []
                self.idx = {0 : 0, 1 : 0} 
                # Reset initial observation to a random x and theta
                if mode == 1:
                    #Validation set
                    ranges_x = [(-0.75,-0.5),(0.5,0.75)]
                    ranges_theta = [(-3*PI/2,-PI/2),(PI/2,3*PI/2)]
                    
                else:
                    #Test set
                    ranges_x = [(-1,-0.75),(0.75,1)]
                    ranges_theta = [(-PI,-PI/2),(PI/2,PI)]
                range_x = ranges_x[self._rng.randint(len(ranges_x))]
                range_theta = ranges_theta[self._rng.randint(len(ranges_theta))]
                for _ in range(self.n_init_states):
                    x = self._rng.uniform(*range_x)
                    theta = self._rng.uniform(*range_theta)
                    self.init_states[mode].append((x,theta))
            
            x = self._rng.uniform(*self.init_states[mode][self.idx[mode]])
            theta = self._rng.uniform(*self.init_states[mode][self.idx[mode]])
            self.idx[mode] = (self.idx[mode] + 1) % self.n_init_states
            
        else:

            #Learning set 
            x = self._rng.uniform(-0.5, 0.5)
            theta = self._rng.uniform(-PI/2, PI/2)
            
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

    def inTerminalState(self):
        return np.any(self._last_observation == np.nan)

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        # The environment allows two different actions to be taken
        # at each time step
        return self._n_actions  

    def observe(self):
        return self.convert_repr() 

if __name__ == "__main__":
    rng = np.random.RandomState(12345)
    env = Cartpole(rng,stepsize=10)
    env.reset(mode=1)
    dataset = DataSet(env)
    
    act = 1
    for i in range(200):
        r = env.act(act)
        obs = env.observe()
        dataset.addSample(obs, act, r, False, 0)
    env.summarizePerformance(dataset,"./test/","testoide")
