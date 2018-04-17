"""
Neural network using Keras (called by q_net_keras) designed for Normalized Advantage Function algorithm 
.. Author: Samy Aittahar 
   Taken from https://github.com/tambetm/gymexperiments/blob/master/naf.py
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, LeakyReLU, Reshape, Permute, concatenate, PReLU, ELU, ThresholdedReLU, Softmax, Lambda, Dropout, BatchNormalization, GaussianDropout
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error, kullback_leibler_divergence
from keras import backend as K
from copy import deepcopy
from keras.regularizers import l2, l1, Regularizer
import theano.tensor as T
lays = dict()
lays["dense"] = Dense

activations = dict()
activations["leakyrelu"] = LeakyReLU
activations["prelu"] = PReLU
activations["elu"] = ELU
activations["thresholdedrelu"] = ThresholdedReLU
activations["softmax"] = Softmax
activations["batchnorm"] = BatchNormalization

def _L(len_u):
    if len_u == 1:
        return lambda x : K.exp(x)
    else:
        def _subL(x):
            # initialize with zeros
            batch_size = x.shape[0]
            a = T.zeros((batch_size, num_actuators, num_actuators))
            # set diagonal elements
            batch_idx = T.extra_ops.repeat(T.arange(batch_size), num_actuators)
            diag_idx = T.tile(T.arange(num_actuators), batch_size)
            b = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(T.exp(x[:, :num_actuators])))
            # set lower triangle
            cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in range(num_actuators)])
            rows = np.concatenate([np.array([i]*i, dtype=np.uint) for i in range(num_actuators)])
            cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
            rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
            batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
            c = T.set_subtensor(b[batch_idx, rows_idx, cols_idx], T.flatten(x[:, num_actuators:]))
        return _subL

def _P(len_u):
    if len_u == 1:
        return lambda x : x*x
    else:
        return lambda x : K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

def _A(len_u):
    if len_u == 1:
        def _subA(x):
            m,p,u = x
            return -(1/2.0)*((u-m) * (u-m)) * p 
    else:
        def _subA(x):
            m, p, u = x
            d = K.expand_dims(u - m, -1)
            return -(1/2.0)*K.batch_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)
    return _subA

def _Q(len_u):
    def _subQ(x):
        v, a = x
        return v + a
    return _subQ


class Naf_nn():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    action_as_input : Boolean
        Whether the action is given as input or as output
    """
    def __init__(self, batch_size=32, input_dimensions=[], n_actions=2, random_state=np.random.RandomState(), layers = [{"type" : "dense", "activation" : "leakyrelu", "activation_kwargs" : {}, "units" : 100},{"type" : "dense", "activation" : "leakyrelu", "activation_kwargs" : {}, "units" : 100}]): 
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._layers = layers

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        global lays, activations
        layers=[]
        outs_conv=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a FRAME
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)
                reshaped=Permute((2,3,1), input_shape=(dim[0],dim[1],dim[2]))(input)    #data_format='channels_last'
                x = Conv2D(8, (4, 4), activation='relu', padding='valid')(reshaped)   #Conv on the frames
                x = Conv2D(16, (3, 3), activation='relu', padding='valid')(x)         #Conv on the frames
                x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)
                x = Conv2D(16, (3, 3), activation='relu', padding='valid')(x)         #Conv on the frames
                
                out = Flatten()(x)
                
            # - observation[i] is a VECTOR
            elif len(dim) == 2:
                if dim[0] > 3:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
                    reshaped=Reshape((dim[0],dim[1],1), input_shape=(dim[0],dim[1]))(input) 
                    x = Conv2D(16, (2, 1), activation='relu', padding='valid')(reshaped)#Conv on the history
                    x = Conv2D(16, (2, 2), activation='relu', padding='valid')(x)       #Conv on the history & features

                    out = Flatten()(x)
                else:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
                    out = Flatten()(input)

            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    # this returns a tensor
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    reshaped=Reshape((1,dim[0],1), input_shape=(dim[0],))(input)  
                    x = Conv2D(8, (1,2), activation='relu', padding='valid')(reshaped)  #Conv on the history
                    x = Conv2D(8, (1,2), activation='relu', padding='valid')(x)         #Conv on the history
                    
                    out = Flatten()(x)
                                        
                else:
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    out=input
                    
            outs_conv.append(out)

        
        if ( isinstance(self._n_actions,int)):
            raise Exception("Error, env.nActions() must be a continuous set in the Naf NN")

        x_input = list(inputs)
        u_input = Input(shape=(len(self._n_actions),))
        xu_input = x_input + [u_input]
        #inputs.append(u_input)
        #outs_conv.append(u_input)
        
        if len(outs_conv)>1:
            x = concatenate(outs_conv)
        else:
            x= outs_conv [0]
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25, seed=self._random_state.seed())(x)#
        # we stack a deep fully-connected network on top
        #x = GaussianDropout(rate=0.1)(x)
        for l in self._layers : 
            x = lays[l["type"]](l["units"])(x)
            
            try : 
                x = Activation(l["activation"],**l["activation_kwargs"])(x)
            except:
                try:
                    x = activations[l["activation"]](**l["activation_kwargs"])(x)
                except:
                    print("Warning : the activation layer you have requested is not available. Activation relu will be used instead.")
                    x = Activation("relu")(x)
            #x = BatchNormalization()(x)
            #x = Dropout(rate=0.5,seed=self._random_state.seed())(x)#
            #x = GaussianDropout(rate=0.25)(x)
        len_u = len(self._n_actions)
        v = Dense(1)(x)
        m = Dense(len_u)(x)
        l0 = Dense(len_u*(len_u+1) // 2)(x)

        l = Lambda(_L(len_u),output_shape=(len_u,len_u))(l0)
        p = Lambda(_P(len_u),output_shape=(len_u,len_u))(l)
        a = Lambda(_A(len_u),output_shape=(len_u,))([m,p,u_input])
        q = Lambda(_Q(len_u),output_shape=(len_u,))([v,a])


        fmu = K.function([*x_input], [m])
        mu = lambda x: fmu([*x])

        fP = K.function([*x_input], [p])
        P = lambda x: fP([*x])

        fA = K.function([*xu_input], [a])
        A = lambda xu: fA([*xu])

        fQ = K.function([*xu_input], [q])
        Q = lambda xu: fQ([*xu])

        fV = K.function([*x_input], [v])
        V = lambda x: fV([*x])
        """
        if (self._action_as_input==False):
            if ( isinstance(self._n_actions,int)):
                out = Dense(self._n_actions)(x)
            else:
                out = Dense(len(self._n_actions))(x)
        else:
            out = Dense(1)(x)
        """
        old_Q = Input(shape=(len_u,))     
        model = Model(inputs=xu_input + [old_Q], outputs=q)
        layers=model.layers
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
        
        return model, params, {"mu":mu,"P":P,"A":A,"Q":Q,"V":V,"oldQ":old_Q}


if __name__ == '__main__':
    pass
    
