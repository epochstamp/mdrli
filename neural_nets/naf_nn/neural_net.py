"""
Neural network using Keras (called by q_net_keras) designed for Normalized Advantage Function algorithm 
.. Author: Samy Aittahar 
   Taken from https://github.com/tambetm/gymexperiments/blob/master/naf.py
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, LeakyReLU, Reshape, Permute, concatenate, PReLU, ELU, ThresholdedReLU, Softmax, Lambda, Dropout, BatchNormalization, GaussianDropout
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error, kullback_leibler_divergence, logcosh
from keras import backend as K
from copy import deepcopy
from keras.regularizers import l2, l1, Regularizer
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
            a = T.zeros((batch_size, len_u, len_u))
            # set diagonal elements
            batch_idx = T.extra_ops.repeat(T.arange(batch_size), len_u)
            diag_idx = T.tile(T.arange(len_u), batch_size)
            b = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(T.exp(x[:, :len_u])))
            # set lower triangle
            cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in range(len_u)])
            rows = np.concatenate([np.array([i]*i, dtype=np.uint) for i in range(len_u)])
            cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
            rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
            batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
            c = T.set_subtensor(b[batch_idx, rows_idx, cols_idx], T.flatten(x[:, len_u:]))
        return _subL

def _P(len_u):
    if len_u == 1:
        return lambda x : x*x
    else:
        return lambda x : K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

def _KLN(len_u):
    if len_u == 1 : 
        def _subKLN(x):
            mu2,sigma2,mu1,sigma1 = x
            return K.log(sigma1/sigma2) + ((sigma1 * sigma1 + ((mu1 - mu2) * (mu1 - mu2))) / (2*sigma2*sigma2)) - 0.5
    else:
        def _subKLN(x):
            mu1,sigma1,mu2,sigma2 = x
            size_sigma1 = K.int_shape(sigma1)[1]
            size_sigma2 = K.int_shape(sigma2)[1]
            inv_sigma2 = K.tf.matrix_inverse(K.tf.diag(sigma2))
            return 0.5 * (K.log(size_sigma1 / size_sigma2) - len_u + K.tf.linalg.trace(K.tf.matmul(inv_sigma2, sigma1)) + K.tf.matmul(K.tf.matmul(K.tf.transpose(mu2 - mu1), inv_sigma2), (mu2-mu1)))
    return _subKLN

def _CL(len_u):
    def _subCL(x):
  
       q,next_q,kldiv = x#kldiv, pred_q, next_q, p = x 
       softkldiv = kldiv/(1+K.abs(kldiv))
       return K.sqrt(softkldiv*softkldiv) *logcosh(q,next_q) # + K.epsilon()*kld
    return _subCL

def _NM(len_u):
    if len_u == 1:
        def _subNM(x):
            m,s = x
            return K.random_normal(shape=(1,), mean=m, stddev=K.sqrt(s))
    else:
        def _subNM(x):
            return K.map_fn(lambda i : K.random_normal_variable(shape=(1,), mean=m[i], scale=s[i]), K.arange(K.constant(len_u)))
    return _subNM

def _S(len_u,c):
    if len_u == 1:
        return lambda x : K.constant(c)/x
    else:
        def _subS(x):
            a = K.constant(c)*K.matrix_inverse(x)
            return a
    return _subS

def _A(len_u):
    if len_u == 1:
        def _subA(x):
            m,p,u = x
            return -(0.5)*((u-m) * (u-m)) * p 
    else:
        def _subA(x):
            m, p, u = x
            d = K.expand_dims(u - m, -1)
            return -(0.5)*K.batch_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)
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
    def __init__(self, batch_size=32, input_dimensions=[], n_actions=2, random_state=np.random.RandomState(), c=0.7,layers = [{"type" : "dense", "activation" : "leakyrelu", "activation_kwargs" : {}, "units" : 200},{"type" : "dense", "activation" : "leakyrelu", "activation_kwargs" : {}, "units" : 200}]): 
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._layers = layers
        self.c = c

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
        x = BatchNormalization()(x)
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
            x = BatchNormalization()(x)
            #x = Dropout(rate=0.5,seed=self._random_state.seed())(x)#
            #x = GaussianDropout(rate=0.25)(x)
        len_u = len(self._n_actions)
        v = Dense(1)(x)
        #v = BatchNormalization()(v)
        m = Dense(len_u)(x)
        #m = BatchNormalization()(m)
        l0 = Dense(len_u*(len_u+1) // 2)(x)
        #l0 = BatchNormalization()(l0)

        next_Q = Input(shape=(len_u,))
        
        old_mu = Input(shape=(len_u,))
        old_sigma = Input(shape=(len_u,))

        l = Lambda(_L(len_u),output_shape=(len_u,len_u))(l0)
        p = Lambda(_P(len_u),output_shape=(len_u,len_u))(l)
        a = Lambda(_A(len_u),output_shape=(len_u,))([m,p,u_input])
        q = Lambda(_Q(len_u),output_shape=(len_u,))([v,a])
        sigma = Lambda(_S(len_u,self.c),output_shape=(len_u,))([p])
        noisymu = Lambda(_NM(len_u), output_shape=(len_u,))([m,sigma])
        

        fmu = K.function([*x_input], [m])
        mu = lambda x: fmu([*x])

        fsigma = K.function([*x_input], [sigma])
        S = lambda x: fsigma([*x])

        fnoisymu = K.function([*x_input], [m,sigma])
        NM = lambda x: fnoisymu([*x])

        fP = K.function([*x_input], [p])
        P = lambda x: fP([*x])

        fA = K.function([*xu_input], [a])
        A = lambda xu: fA([*xu])

        fQ = K.function([*xu_input], [q])
        Q = lambda xu: fQ([*xu])

        fV = K.function([*x_input], [v])
        V = lambda x: fV([*x])

        

        divergence_norm = Lambda(_KLN(len_u), output_shape=(len_u,))([old_mu,old_sigma,m,sigma])
        custom_loss = Lambda(_CL(len_u), output_shape=(len_u,))([q, next_Q,divergence_norm])

        #fCL = K.function([*x_input], [custom_loss])
        #cl = lambda x : fCL([*x])

        #fKLN = K.function([*x_input], [divergence_norm])
        #KLN = lambda x : fKLN([*x])
        """
        if (self._action_as_input==False):
            if ( isinstance(self._n_actions,int)):
                out = Dense(self._n_actions)(x)
            else:
                out = Dense(len(self._n_actions))(x)
        else:
            out = Dense(1)(x)
        """
        
        #old_prob = Input(shape=(len_u,))    
        model = Model(inputs=xu_input + [next_Q,old_mu,old_sigma], outputs=custom_loss)
        layers=model.layers
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
        
        return model, params, {"mu":mu,"P":P,"A":A,"Q":Q,"V":V,"noisymu" : NM, "sigma":S, "loss" :lambda y_true,y_pred : y_pred }


if __name__ == '__main__':
    pass
    
