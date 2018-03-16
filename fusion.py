from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, concatenate, LeakyReLU
from keras.models import Model
import numpy as np
def fusion(net1,net2,input_shape,output_size):
    main_input = Input(shape=(1,) + input_shape, dtype='float32', name='main_input')
    #interface1 = Dense(2,activation='linear')(main_input)
    #interface2 = Dense(2,activation='linear')(main_input)
    main_output1 = net1(main_input)
    main_output2 = net2(main_input)
    combined_output = concatenate([main_output2,main_output1])
    
    selections = []
    outputs = []
    for s in range(0,output_size):
        #Selection 1
        selection_1_inputs = Input(shape=(2*output_size,))
        selection_1 = Dense(1, activation='linear')(selection_1_inputs)
        weighs = []
        for k in range(0,2):
            for i in range(0,output_size):
                if(i==s):
                    weighs.append([0.5])
                else:
                    weighs.append([0])
                    
        selection_1_weigths = [np.array(weighs, dtype="f"), np.array([0], dtype="f")]
        selection_1_model = Model(inputs=selection_1_inputs, outputs=selection_1, name='SL'+str(s))
        selection_1_model.layers[1].set_weights(selection_1_weigths)
        selection_1_model.compile(optimizer='sgd', loss='mse')
        
        selections.append(selection_1_model)
        outputs.append(selection_1_model(combined_output))
    final_output = concatenate(outputs)
    final_model = Model(inputs=[main_input], outputs=[final_output])
    final_model.compile(optimizer='sgd', loss='mse')
    return final_model
            