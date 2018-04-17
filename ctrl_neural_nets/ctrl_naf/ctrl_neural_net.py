"""
Code for general deep Q-learning using Keras that can take as inputs scalars, vectors and matrices

.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.optimizers import SGD,RMSprop, Adam
from keras import backend as K
import keras.losses
from keras.losses import mean_squared_error, kullback_leibler_divergence, logcosh
from ctrl_neural_nets.ctrl_neural_net import QNetwork
from deer.q_networks.NN_keras import NN # Default Neural network used
from keras.models import load_model
from joblib import dump,load
import os
from copy import deepcopy

#def cross_covariance(x,y):
    

def weighted_mse_kl_divergence(old_q):
    def loss(y_true,y_pred):
        #y_true = K.clip(y_true,K.epsilon(),1)
        #y_pred = K.clip(y_pred,K.epsilon(),1)
        rt = K.mean((K.softmax(old_q)/K.softmax(y_pred)))
        c = K.clip(rt,0.8,1.2)
        return K.minimum(rt,c) * y_true + mean_squared_error(y_true,y_pred)
    return loss


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

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_value=0,clip_norm=0.0, freeze_interval=1000, tau=0.001, batch_size=32, update_rule="adam", random_state=np.random.RandomState(), double_Q=False, neural_network=NN, neural_network_kwargs={}, loss='mse'):
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
        self.tau = tau
        if "input_dimensions" not in neural_network_kwargs:
            neural_network_kwargs["input_dimensions"] = self._input_dimensions
        if "n_actions" not in neural_network_kwargs:
            neural_network_kwargs["n_actions"] = self._n_actions

        Q_net = neural_network(batch_size=self._batch_size, random_state=self._random_state,**neural_network_kwargs)

        

        self.q_naf, self.params, self.naf_components = Q_net._buildDQN()
       
        self._noise_functor = None
        self._loss = loss
        self._compile(weighted_mse_kl_divergence,old_q=self.naf_components["oldQ"])
        
        self.next_q_naf, self.next_params, self.next_naf_components = Q_net._buildDQN()

        self.V_next = self.next_naf_components["V"]
        self.mu_next = self.next_naf_components["mu"]
        self.P = self.naf_components["P"]
        self.mu = self.naf_components["mu"]
        self.Q = self.naf_components["Q"] 
        self.V = self.naf_components["V"]
        self.oldQ = self.naf_components["oldQ"]
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
            print("coucou")
            self._resetQHat()
        
         
        #next_u = self.mu_next(next_states_val.tolist())
        #next_q_naf = self.next_q_naf.predict(np.concatenate(next_states_val.tolist(), next_u))
        
        """
        if(self._double_Q==True):
            next_q_vals_current_qnet=self.q_vals.predict(next_states_val.tolist())
            argmax_next_q_vals=np.argmax(next_q_vals_current_qnet, axis=1)
            max_next_q_vals=next_q_vals[np.arange(self._batch_size),argmax_next_q_vals].reshape((-1, 1))
        else:
            max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)
        """
        #Normalize actions
        k=0
        max_new = 1
        min_new = -1
        """
        print("Before transform")
        print (actions_val)
        for a in self._n_actions:
            max_old = max(a[1], np.amax(actions_val[:,k]))
            min_old = min(a[0], np.amin(actions_val[:,k]))
            
            actions_val[:,k] = ((max_new-min_new)/(max_old-min_old)) * (actions_val[:,k] - max_old) + max_new 
            k+=1
        print("After transform")
        """
        #print (actions_val)

        tau = self.tau
        not_terminals=np.ones_like(terminals_val) ^ terminals_val
        v1 = self.V_next(next_states_val)
        v2 = self.V(next_states_val)
        s = np.array(states_val.tolist())
        a = np.expand_dims(np.array(actions_val.tolist()),axis=0)
        sa = np.concatenate((s,a), axis=0)
        v = ((1-tau) * np.array(v1) + (tau) * np.array(v2))
        qsa = self.Q(sa)
        #print(qsa)
        y = rewards_val + not_terminals * self._df * np.squeeze(v)
        

        
        
        #print (np.stack((states_val.tolist(),actions_val.tolist())))
        #print(np.concatenate(np.array(states_val.tolist()),actions_val.reshape((-1,))).shape)
        q_vals=np.squeeze(np.array(qsa))


        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        #q_val=q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]#.reshape((-1, 1))
        #q_val = q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]      
        diff = y - q_vals
        loss_ind=diff*diff
                
        # Is it possible to use something more flexible than this? 
        # Only some elements of next_q_vals are actual value that I target. 
        # My loss should only take these into account.
        # Workaround here is that many values are already "exact" in this update
        states_actions = states_val.tolist()
        states_actions.append(actions_val)
        states_actions.append(np.clip(np.array(np.squeeze(qsa)),K.epsilon(),1))
        loss = 0
        k = 1
        for _ in range(k):
            l = self.q_naf.train_on_batch(states_actions , y )   
            #print(l)
            loss+= (1/float(k)) * l  
        #print("Loss = " + str(loss))
        self.update_counter += 1        
        # loss*self._n_actions = np.average(loss_ind)
 
        #Update target weights
        
        
        w = self.q_naf.get_weights()
        next_w = self.next_q_naf.get_weights()
        for i in range(len(w)):
            next_w[i] = tau * w[i] + (1-tau) * next_w[i]
        self.next_q_naf.set_weights(next_w)
        
        return np.sqrt(loss),loss_ind

    def batchPredict(self, states_val):
        return self.V(states_val)


    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """   

        s = [np.expand_dims(s,axis=0) for s in state]
        mus = self.mu(s)
        u = mus[0][0]
        s.append(mus[0])
        q = self.Q(s)
        return u,q

    def evalAction(self, state, action):
        """ Evaluate an action

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """     
        s = [np.expand_dims(s,axis=0) for s in state]


        s.append([np.asarray(action)])
        q = self.Q(s)
        return q

    def getCopy(self):
        """Return a copy of the current
        """
        self.dumpTo()
        copycat = deepcopy(self)
        copycat.load()
        self.load()
        return copycat
        
        
    def _compile(self,loss=None,**lkwargs):
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
        elif (self._update_rule=="adam"):
            optimizer = Adam(lr=self._lr,**kwargs)
        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        if loss is None:
            loss = self._loss
        self.q_naf.compile(optimizer=optimizer, loss=loss if isinstance(loss,str) else loss(**lkwargs))
        #self.next_q_naf.compile(optimizer=optimizer, loss=loss if isinstance(loss,str) else loss(*args,**kwargs))
    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            K.set_value(next_param,K.get_value(param))

        self._compile(weighted_mse_kl_divergence,old_q=self.naf_components["oldQ"]) # recompile to take into account new optimizer parameters that may have changed since
                        # self._compile() was called in __init__. FIXME: this call should ideally be done elsewhere
