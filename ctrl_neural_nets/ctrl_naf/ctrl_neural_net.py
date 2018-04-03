"""
Code for general deep Q-learning using Keras that can take as inputs scalars, vectors and matrices

.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.optimizers import SGD,RMSprop
from keras import backend as K
import keras.losses
from keras.losses import mean_squared_error, kullback_leibler_divergence
from ctrl_neural_nets.ctrl_neural_net import QNetwork
from deer.q_networks.NN_keras import NN # Default Neural network used
from keras.models import load_model
from joblib import dump,load
import os
from copy import deepcopy



def noise_function(name,rng,n_actions):
       
    def zero(u,noise_scale):
        return 0

    def linear(u,noise_scale):
        return u + rng.randn(n_actions) / (linear_coeff)

    def exp(u,noise_scale):
        return u + rng.randn(n_actions) * 10 ** (-noise_scale)

    def fixed(u,noise_scale):
        return u + rng.randn(n_actions) * noise_scale

    def covariance(u,noise_scale,p = None):
        if p is None : return 0
        if n_actions == 1:
            std = np.minimum(noise_scale / p(x)[0], 1)
            action = rng.normal(u,std,size=(1,))
        else:
            cov = np.minimum(np.linalg.inv(p(x)[0]) * noise_scale, 1)
            action = np.random.multivariate_normal(u, cov)
        return action

    try:
        return eval(name)
    except Exception as e:
        print("Warning : The name does not refer to a valid mapping between name and noise function. Output of the exception : ")
        print(e)
    

class Ctrl_naf(QNetwork):
    """
    Deep Q-learning network using Keras (with any backend)
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Default : 0
    clip_norm : float
        if clip_norm > 0, all parameters gradient will be clipped to a maximum norm of clip_norm
    clip_value : float
        if clip_norm > 0, all parameters gradient will be clipped to a maximum value of clip_value
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : class, optional
        default is deer.qnetworks.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_value=0,clip_norm=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network=NN, neural_network_kwargs={}, loss='mse'):
        """ Initialize environment
        
        """

        QNetwork.__init__(self,environment, batch_size)
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._update_rule = update_rule
        self._clip_value = clip_value
        self._clip_norm = clip_norm
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0
        if "input_dimensions" not in neural_network_kwargs:
            neural_network_kwargs["input_dimensions"] = self._input_dimensions
        if "n_actions" not in neural_network_kwargs:
            neural_network_kwargs["n_actions"] = self._n_actions

        Q_net = neural_network(batch_size=self._batch_size, random_state=self._random_state,**neural_network_kwargs)

        

        self.q_naf, self.params, self.naf_components = Q_net._buildDQN()
       

        self._loss = loss
        self._compile()
        
        self.next_q_naf, self.next_params, self.next_naf_components = Q_net._buildDQN()

        self.V = self.next_naf_components[-1]
        self.P = self.naf_components[1]
        self.mu = self.naf_components[0]
        self.Q = self.naf_components[-2] 
        self.next_q_naf.set_weights(self.q_naf.get_weights)
        self._resetQHat()

    def getAllParams(self):
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(K.get_value(p))
        return params_value

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])

    
    def dumpTo(self,out=None):
        q_naf = self.q_naf
        next_q_naf = self.next_q_naf
        q_naf.save("temp.h5")
        self.q_naf = open("temp.h5","rb").read()
        next_q_naf.save("temp.h5")
        self.next_q_naf = open("temp.h5","rb").read()
        os.remove("temp.h5")
        #Remove params - we retrieve them later by load method
        self.params = None
        self.next_params = None
        self.dumped = True
        if out is not None:
            dump(self,out)

    def load(self):
        if hasattr(self,"dumped") and self.dumped:
            f = open("temp.h5","wb")
            f.write(self.q_naf)
            f.close()
            self.q_naf = load_model("temp.h5")
            f = open("temp.h5","wb")
            f.write(self.next_q_naf)
            f.close()
            self.next_q_naf = load_model("temp.h5")
            layers=self.q_naf.layers
            
            # Get back params
            self.params = [ param
                        for layer in layers 
                        for param in layer.trainable_weights ]
            layers=self.next_q_naf.layers
            self.next_params = [ param
                        for layer in layers 
                        for param in layer.trainable_weights ]
            

    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.

        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training

        Parameters
        -----------
        states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val : b x 1 numpy array of integers
        rewards_val : b x 1 numpy array
        next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val : b x 1 numpy boolean array

        Returns
        -------
        Average loss of the batch training (RMSE)
        Individual (square) losses for each tuple
        """
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        
        next_q_naf = self.next_q_naf.predict(next_states_val.tolist())
        """
        if(self._double_Q==True):
            next_q_vals_current_qnet=self.q_vals.predict(next_states_val.tolist())
            argmax_next_q_vals=np.argmax(next_q_vals_current_qnet, axis=1)
            max_next_q_vals=next_q_vals[np.arange(self._batch_size),argmax_next_q_vals].reshape((-1, 1))
        else:
            max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)
        """
        not_terminals=np.ones_like(terminals_val) ^ terminals_val
        v = self.V(next_states)
        
        y = rewards_val + not_terminals * self._df * np.squeeze(v)
        
        q_vals=self.q([states_val.tolist(), actions_val.tolist()])

        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        #q_val=q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]#.reshape((-1, 1))
        q_val = q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]      
        diff = - q_val + target 
        loss_ind=pow(diff,2)
                
        q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ] = target
                
        # Is it possible to use something more flexible than this? 
        # Only some elements of next_q_vals are actual value that I target. 
        # My loss should only take these into account.
        # Workaround here is that many values are already "exact" in this update
        loss=self.q_vals.train_on_batch([states_val.tolist(), actions_val.tolist()] , q_vals )      
        self.update_counter += 1        
        # loss*self._n_actions = np.average(loss_ind)
        return np.sqrt(loss),loss_ind

    def batchPredict(self, states_val):
        return self.q_vals.predict(states_val.tolist())


    def chooseBestAction(self, state,noise_func="zero", noise_coeff=0):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """   
        if self._noise_functor is None : self._noise_functor = noise_function(noise_func,self._random_state,self._n_actions)     
        kwargs = {"p" : self.P}
        try:
            u = self._noise_functor(self.mu(state)[0],noise_coeff,**kwargs)
        except:
            u = self._noise_functor(self.mu(state)[0],noise_coeff)
        q = self.Q(state,u)
        return u,q

    def getCopy(self):
        """Return a copy of the current
        """
        self.dumpTo()
        copycat = deepcopy(self)
        copycat.load()
        self.load()
        return copycat
        

    def updateLossFunction(self,loss=None,*args,**kwargs):
        self._compile(loss,*args,**kwargs)
        
    def _compile(self,loss=None,*args,**kwargs):
        """ compile self.q_vals
        """
        global loss_functions
        kwargs = dict()
        if self._clip_value > 0:
            kwargs["clipvalue"] = self._clip_value
        if self._clip_norm > 0:
            kwargs["clipnorm"] = self._clip_norm
        if (self._update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False,**kwargs)
        elif (self._update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon,**kwargs)
        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        
        if loss is None:
            loss = self._loss
        elif loss in loss_functions.keys():
            loss = loss_functions[loss]
        self.q_naf.compile(optimizer=optimizer, loss=loss if isinstance(loss,str) else loss(*args,**kwargs))

    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            K.set_value(next_param,K.get_value(param))

        self._compile() # recompile to take into account new optimizer parameters that may have changed since
                        # self._compile() was called in __init__. FIXME: this call should ideally be done elsewhere
