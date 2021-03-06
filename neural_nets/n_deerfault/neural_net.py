"""
Neural network using Keras (called by q_net_keras)
.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, LeakyReLU, Reshape, Permute, concatenate, PReLU, ELU, ThresholdedReLU, Softmax
lays = dict()
lays["dense"] = Dense

activations = dict()
activations["leakyrelu"] = LeakyReLU
activations["prelu"] = PReLU
activations["elu"] = ELU
activations["thresholdedrelu"] = ThresholdedReLU
activations["softmax"] = Softmax



class N_deerfault():
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
    def __init__(self, batch_size=32, input_dimensions=[], n_actions=2, random_state=np.random.RandomState(), action_as_input=False, layers = [{"type" : "dense", "activation" : "leakyrelu", "activation_kwargs" : {}, "units" : 50},{"type" : "dense", "activation" : "elu", "activation_kwargs" : {}, "units" : 20}]): 
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._action_as_input=action_as_input
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

        if (self._action_as_input==True):
            if ( isinstance(self._n_actions,int)):
                print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
            else:
                input = Input(shape=(len(self._n_actions),))
                inputs.append(input)
                outs_conv.append(input)
        
        if len(outs_conv)>1:
            x = concatenate(outs_conv)
        else:
            x= outs_conv [0]
        
        # we stack a deep fully-connected network on top

        for l in self._layers : 
            x = lays[l["type"]](l["units"])(x)
            try : 
                x = Activation(l["activation"],**l["activation_kwargs"])(x)
            except:
                try:
                    x = activations[l["activation"]](**l["activation_kwargs"])(x)
                except:
                    print("Warning : the activation layer you have requested is not available. Activation relu will be used")
                    x = Activation("relu")(x)

        #x = Dense(50, activation='elu')(x)
        #x = Dense(20, activation='elu')(x)
        #x = Dense(20)(x)
        #x = LeakyReLU(alpha=0.3)(x)
        #x = Dense(20)(x)
        #x = LeakyReLU(alpha=0.3)(x)
        #x = Dense(20)(x)
        #x = LeakyReLU(alpha=0.3)(x)
        #x = Dense(20)(x)
        #x = LeakyReLU(alpha=0.3)(x)
       
        if (self._action_as_input==False):
            if ( isinstance(self._n_actions,int)):
                out = Dense(self._n_actions)(x)
            else:
                out = Dense(len(self._n_actions))(x)
        else:
            out = Dense(1)(x)
                
        model = Model(inputs=inputs, outputs=out)
        layers=model.layers
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
        
        if (self._action_as_input==True):
            return model, params, inputs
        else:
            return model, params

if __name__ == '__main__':
    pass
    
