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

sys.path.insert(0, 'utils/doublecartpole')
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'envs/')
sys.path.insert(0, '.')
from render_movie import save_mp4
from env import Environment
from data.dataset import DataSet
from scipy.integrate import RK45

# Physics constants
PI = np.pi

N_TAU=10
DELTA_T=0.02

def rK6(a, b, c, d, e, f, fa, fb, fc, fd, fe, ff, hs):
	a1 = fa(a, b, c, d, e, f)*hs
	b1 = fb(a, b, c, d, e, f)*hs
	c1 = fc(a, b, c, d, e, f)*hs
	d1 = fd(a, b, c, d, e, f)*hs
	e1 = fe(a, b, c, d, e, f)*hs
	f1 = ff(a, b, c, d, e, f)*hs
	ak = a + a1*0.5
	bk = b + b1*0.5
	ck = c + c1*0.5
	dk = d + d1*0.5
	ek = e + e1*0.5
	fk = f + f1*0.5
	a2 = fa(ak, bk, ck, dk, ek, fk)*hs
	b2 = fb(ak, bk, ck, dk, ek, fk)*hs
	c2 = fc(ak, bk, ck, dk, ek, fk)*hs
	d2 = fd(ak, bk, ck, dk, ek, fk)*hs
	e2 = fe(ak, bk, ck, dk, ek, fk)*hs
	f2 = ff(ak, bk, ck, dk, ek, fk)*hs
	ak = a + a2*0.5
	bk = b + b2*0.5
	ck = c + c2*0.5
	dk = d + d2*0.5
	ek = e + e2*0.5
	fk = f + f2*0.5
	a3 = fa(ak, bk, ck, dk, ek, fk)*hs
	b3 = fb(ak, bk, ck, dk, ek, fk)*hs
	c3 = fc(ak, bk, ck, dk, ek, fk)*hs
	d3 = fd(ak, bk, ck, dk, ek, fk)*hs
	e3 = fe(ak, bk, ck, dk, ek, fk)*hs
	f3 = ff(ak, bk, ck, dk, ek, fk)*hs
	ak = a + a3
	bk = b + b3
	ck = c + c3
	dk = d + d3
	ek = e + e3
	fk = f + f3
	a4 = fa(ak, bk, ck, dk, ek, fk)*hs
	b4 = fb(ak, bk, ck, dk, ek, fk)*hs
	c4 = fc(ak, bk, ck, dk, ek, fk)*hs
	d4 = fd(ak, bk, ck, dk, ek, fk)*hs
	e4 = fe(ak, bk, ck, dk, ek, fk)*hs
	f4 = ff(ak, bk, ck, dk, ek, fk)*hs
	a = a + (a1 + 2*(a2 + a3) + a4)/6
	b = b + (b1 + 2*(b2 + b3) + b4)/6
	c = c + (c1 + 2*(c2 + c3) + c4)/6
	d = d + (d1 + 2*(d2 + d3) + d4)/6
	e = e + (e1 + 2*(e2 + e3) + e4)/6
	f = f + (f1 + 2*(f2 + f3) + f4)/6
	return a, b, c, d, e, f



    

class Doublecartpole(Environment):
    def __init__(self, rng, g=9.8,m_cart=1.5,m_pole_1=0.5,m_pole_2=0.75,l1=0.5,l2=0.75,min_f=-1,max_f=1,r=0,s=5,stepsize=2):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        # Defining the type of environment
        self._rng = rng
        # Observations = (x, x_dot, theta, theta_dot, timestamp)
        #self._last_observation = [0, 0, 0, 0]
        self._input_dim = [(1,), (1,), (1,), (1,),(1,),(1,)]
        self._video = 0
        self.g = float(g)  
        self.m_cart = float(m_cart)
        self.m_pole_1 = float(m_pole_1)
        self.m_pole_2 = float(m_pole_2)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.r = float(r)
        self.s = float(s)
        self.min_f = min_f
        self.max_f = max_f
        self._continuous = stepsize <= 0
        self._n_actions = [[self.min_f,self.max_f]] if self._continuous else int((self.max_f - self.min_f)/stepsize) + 1
        self.stepsize = stepsize
        self.actions = None
            
           
            

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
            force = action[0]
        else:
            if self.actions is None:    
                self.actions = np.arange(self.min_f, self.max_f + self.stepsize, self.stepsize)
            force = self.actions[action]
        # Divide DELTA_T into smaller tau's, to better take into account
        # the transitions
        n_tau = N_TAU
        delta_t = DELTA_T

        tau = delta_t / n_tau
        #print ("Before calculation")
        #print (self._last_observation)
        m1 = self.m_pole_1
        m2 = self.m_pole_2
        m = self.m_cart
        L1 = self.l1
        L2 = self.l2
        g = self.g
        #Intermediate values
        #l1 = L1/2.0
        #l2 = L2/2.0
        #I1 = m1 * ((L1*L1)/12.0)
        #I2 = m2 * ((L2*L2)/12.0)
        d1 = m + m1 + m2
        d2 = ((m1/2.0) + m2)*L1
        d3 = (m2*L2)/2.0 
        d4 = ((1/3.0 * m1) + m2)*L1*L1
        d5 = (1/2.0)*(m2*L1*L2)
        d6 = (1/3.0) * m2*L2*L2
        f1 = (((1/2.0) * m1) + m2)*L1*g
        f2 = (1/2.0)*m2*L2*g
        #d1 = self.d1
        #d2 = self.d2
        #d3 = self.d3
        for i in range(n_tau):
            
            # Physics -> See wiki for the formulas
            args = tuple(self._last_observation)
            def s(x,i):
                x, x_dot, theta_1, theta_1_dot,theta_2, theta_2_dot = self._last_observation#_ = self._last_observation
                cos_theta_1 = np.cos(theta_1)
                cos_theta_2 = np.cos(theta_2)
                sin_theta_1 = np.sin(theta_1)
                sin_theta_2 = np.sin(theta_2)
                cos_diff_theta = np.cos(theta_1 - theta_2)
                sin_diff_theta = np.cos(theta_1 - theta_2)
                
                D = np.asarray([[d1, d2*cos_theta_1, d3*cos_theta_2],
                                [d2*cos_theta_1, d4, d5*cos_diff_theta],
                                [d3*cos_theta_2, d5*cos_diff_theta, d6]])

                C = np.asarray([[0, -d2*sin_theta_1*theta_1_dot, -d3*sin_theta_2*theta_2_dot],
                                    [0, 0, d5*sin_diff_theta*theta_2_dot],
                                    [0, -d5*sin_diff_theta*theta_1_dot, 0]])
     
                G = np.asarray([[0],
                                    [-f1*sin_theta_1],
                                    [-f2*sin_theta_2]])

                Hu = np.asarray([force, 0, 0])

   
            
                lhs_mat = D
                print((C*np.asarray([x_dot,theta_1_dot,theta_2_dot])))               
                rhs_mat = Hu-(C*np.asarray([x_dot,theta_1_dot,theta_2_dot]))-G
                return np.linalg.solve(lhs_mat,rhs_mat)[i]
                
            args += (lambda a,b,c,d,e,f : b, lambda a,b,c,d,e,f : s((a,b,c,d,e,f),0)[0],lambda a,b,c,d,e,f : d, lambda a,b,c,d,e,f : s((a,b,c,d,e,f),1)[0], lambda a,b,c,d,e,f : f, lambda a,b,c,d,e,f : s((a,b,c,d,e,f),2)[0]) + (tau,) 
            self._last_observation = list(rK6(*args))
            self._last_observation[2] = self._to_range(self._last_observation[2])
            self._last_observation[4] = self._to_range(self._last_observation[4])
            """
            
            x, x_dot, theta_1, theta_1_dot,theta_2, theta_2_dot = self._last_observation#_ = self._last_observation
            cos_theta_1 = np.cos(theta_1)
            cos_theta_2 = np.cos(theta_2)
            sin_theta_1 = np.sin(theta_1)
            sin_theta_2 = np.sin(theta_2)
            cos_diff_theta = np.cos(theta_1 - theta_2)
            sin_diff_theta = np.cos(theta_1 - theta_2)


            

            #Intermediate matrices
            D = np.asarray([[d1, d2*cos_theta_1, d3*cos_theta_2],
                                [d2*cos_theta_1, d4, d5*cos_diff_theta],
                                [d3*cos_theta_2, d5*cos_diff_theta, d6]])

            C = np.asarray([[0, -d2*sin_theta_1*theta_1_dot, -d3*sin_theta_2*theta_2_dot],
                                [0, 0, d5*sin_diff_theta*theta_2_dot],
                                [0, -d5*sin_diff_theta*theta_1_dot, 0]])
 
            G = np.asarray([[0],
                                [-f1*sin_theta_1],
                                [-f2*sin_theta_2]])

            Hu = np.asarray([force, 0, 0])

   
            
            lhs_mat = D
                                          
            rhs_mat = Hu-(C*np.asarray([x_dot,theta_1_dot,theta_2_dot]))-G
            #print (rhs_mat)
            sols = np.linalg.solve(lhs_mat,rhs_mat)
            x_dd = sols[0][0]
            theta_1_dd = sols[1][0]
            theta_2_dd = sols[2][0]
            
            #print (theta_1_dd)    
            
            # Update observation vector
            
            self._last_observation = [
                x + tau*x_dot,
                x_dot + tau*x_dd,
                self._to_range(theta_1 + (tau)*theta_1_dot),
                theta_1_dot + tau*theta_1_dd,
                self._to_range(theta_2 + (tau)*theta_2_dot),
                theta_2_dot + tau*theta_2_dd,
                ]
            
            """
            
        #print ("After calculation")
        #print (self._last_observation)
        if (np.isnan(self._last_observation).any()): exit()
        # Simple reward
        #self._last_observation[2] = self._to_range(self._last_observation[2])
        #self._last_observation[4] = self._to_range(self._last_observation[4])
        reward = -abs(self._last_observation[2])
        reward -= abs(self._last_observation[4]) 
        reward -= abs(self._last_observation[0])/2.
        # The cart cannot move beyond -S or S
        S = float(self.s)
        if(self._last_observation[0]<-S):
            self._last_observation[0]=-S
        if(self._last_observation[0]>S):
            self._last_observation[0]=S
 
        return reward

    def convert_repr(self):
        last_observation = copy.deepcopy(self._last_observation)
        #last_observation[2] = self._to_range(self._last_observation[2])
        #last_observation[4] = self._to_range(self._last_observation[4])
        return last_observation
                
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
            theta_2 = self._rng.uniform(-PI/4, PI/4)
        elif mode == 3:
            x = 0
            theta = 0
            theta_2 = 0
        else:
            if mode == 1:
                #Validation set
                ranges_x = [(-0.5,-0.25),(0.25,0.5)]
                ranges_theta = [(-PI/2,-PI/4),(PI/4,PI/2)]
                ranges_theta_2 = [(-PI/2,-PI/4),(PI/4,PI/2)]
            else:
                #Test set
                ranges_x = [(-1,-0.5),(0.5,1)]
                ranges_theta = [(-PI/2,-PI/4),(PI/4,PI/2)]
                ranges_theta_2 = [(-PI/2,-PI/4),(PI/4,PI/2)]
            range_x = ranges_x[self._rng.randint(len(ranges_x))]
            range_theta = ranges_theta[self._rng.randint(len(ranges_theta))]
            range_theta_2 = ranges_theta_2[self._rng.randint(len(ranges_theta))]            

            x = self._rng.uniform(*range_x)
            theta = self._rng.uniform(*range_theta)
            theta_2 = self._rng.uniform(*range_theta_2)
            
        self._last_observation = [x, 0, theta, 0, theta_2, 0]
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
        for i in range(1, 6):
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
        return self._n_actions             

    def observe(self):
        return self.convert_repr() 

if __name__ == "__main__":
    rng = np.random.RandomState()
    env = Doublecartpole(rng,min_f=-5,max_f=5,stepsize=5)
    env.reset(mode=-1)
    dataset = DataSet(env)
    
    print(env.nActions())
    for i in range(1000):
        act = rng.randint(env.nActions())
        r = env.act(act)
        obs = env.observe()
        dataset.addSample(obs, act, r, False, 0)
    env.summarizePerformance(dataset,"./test/","testoide")
