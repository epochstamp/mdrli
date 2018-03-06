"""
Code for general deep Q-learning using Keras that can take as inputs scalars, vectors and matrices

.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.optimizers import SGD,RMSprop
from keras import backend as K
from ctrl_neural_nets.ctrl_neural_net import QNetwork
from deer.q_networks.NN_keras import NN # Default Neural network used
from keras.models import load_model
from joblib import dump,load
import os
from copy import deepcopy


def skillkeeper_loss(w1,w2,y_targ,y_true,y_pred):
        def loss(y_true,y_pred):
                return w1*K.kullback_leibler_divergence(y_true,y_targ) + w2*K.mean_squared_logarithmic_error(y_true,y_pred)
        return loss

class Ctrl_deerfault(QNetwork):
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
    clip_delta : float
        Not implemented.
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
    neural_network : object, optional
        default is deer.qnetworks.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_delta=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """

        QNetwork.__init__(self,environment, batch_size)
        print(double_Q)
        print("yo yo")
        
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._update_rule = update_rule
        #self.clip_delta = clip_delta
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0
                
        try:        
                Q_net = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state)
        except:
                Q_net = neural_network
        self.q_vals, self.params = Q_net._buildDQN()
        
        self._compile()

        self.next_q_vals, self.next_params = Q_net._buildDQN()
        self.next_q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

        self._resetQHat()

    def getAllParams(self):
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(K.get_value(p))
        return params_value

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])
    """       
    def dumpBackEnd(self, out_model, out_params):
        self.q_vals.save(out_model)
        dump(self.getAllParams(), out_params)
        
    def loadBackEndModel(self, in_model,inplace==True):
        if not inplace:
                return load_model(in_model)
        else:
                self.q_vals = load_model(in_model)
    def loadBackEndParams(self, in_params):
        p = load(in_model)
        params_value=[]
        for i,p in enumerate(p):
            params_value.append(K.get_value(p))
        return params_value
    """
    
    def dumpTo(self,out):
        q_vals = self.q_vals
        next_q_vals = self.next_q_vals
        q_vals.save("temp.h5")
        self.q_vals = open("temp.h5","rb").read()
        next_q_vals.save("temp.h5")
        self.next_q_vals = open("temp.h5","rb").read()
        os.remove("temp.h5")
        #Remove params - we retrieve them later by load method
        self.params = None
        self.next_params = None
        self.dumped = True
        dump(self,out)
        
    def load(self):
        if hasattr(self,"dumped") and self.dumped:
            f = open("temp.h5","wb")
            f.write(self.q_vals)
            f.close()
            self.q_vals = load_model("temp.h5")
            f = open("temp.h5","wb")
            f.write(self.next_q_vals)
            f.close()
            self.next_q_vals = load_model("temp.h5")
            layers=self.q_vals.layers
            
            # Get back params
            self.params = [ param
                        for layer in layers 
                        for param in layer.trainable_weights ]
            layers=self.next_q_vals.layers
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
        
        next_q_vals = self.next_q_vals.predict(next_states_val.tolist())
        
        if(self._double_Q==True):
            next_q_vals_current_qnet=self.q_vals.predict(next_states_val.tolist())
            argmax_next_q_vals=np.argmax(next_q_vals_current_qnet, axis=1)
            max_next_q_vals=next_q_vals[np.arange(self._batch_size),argmax_next_q_vals].reshape((-1, 1))
        else:
            max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)

        not_terminals=np.ones_like(terminals_val) ^ terminals_val
        
        target = rewards_val + not_terminals * self._df * max_next_q_vals.reshape((-1))
        
        q_vals=self.q_vals.predict(states_val.tolist())

        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        q_val=q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]#.reshape((-1, 1))        
        diff = - q_val + target 
        loss_ind=pow(diff,2)
                
        q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ] = target
                
        # Is it possible to use something more flexible than this? 
        # Only some elements of next_q_vals are actual value that I target. 
        # My loss should only take these into account.
        # Workaround here is that many values are already "exact" in this update
        loss=self.q_vals.train_on_batch(states_val.tolist() , q_vals )      
        self.update_counter += 1        
        # loss*self._n_actions = np.average(loss_ind)
        return np.sqrt(loss),loss_ind

    def batchPredict(self, states_val):
        return self.q_vals.predict(states_val.tolist())

    def qValues(self, state_val):
        """ Get the q values for one belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q values for the provided belief state
        """ 
        return self.q_vals.predict([np.expand_dims(state,axis=0) for state in state_val])[0]

    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """        
        q_vals = self.qValues(state)

        return np.argmax(q_vals),np.max(q_vals)

    def getCopy():
        """Return a copy of the current
        """
        self.dumpTo()
        copycat = deepcopy(self)
        copycat.load()
        return copycat
        
        
    def _compile(self,loss='mse',*args):
        """ compile self.q_vals
        """
        if (self._update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False)
        elif (self._update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon)
        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        
        self.q_vals.compile(optimizer=optimizer, loss=loss if isinstance(loss,str) else loss(*args))

    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            K.set_value(next_param,K.get_value(param))

        self._compile() # recompile to take into account new optimizer parameters that may have changed since
                        # self._compile() was called in __init__. FIXME: this call should ideally be done elsewhere
