from keras.models import Sequential
from keras.layers import Input, Dense, merge, Activation, Flatten, concatenate, LeakyReLU
from keras.models import Model
import numpy as np
def fusion(net1,net2):
    input_dim = net1._input_dimensions
    output_dim = net1._n_actions
    inputs = []
    for i, dim in enumerate(input_dim):
        input = Input(shape=(dim[0],))
        inputs.append(input)
    #interface1 = Dense(2,activation='linear')(main_input)
    #interface2 = Dense(2,activation='linear')(main_input)
    net1.q_vals.name = "Model_gref_1"
    net2.q_vals.name = "Model_gref_2"
    main_output1 = net1.q_vals(inputs)
    main_output2 = net2.q_vals(inputs)
    combined_output = concatenate([main_output2,main_output1])
    
    selections = []
    outputs = []
    for s in range(0,output_dim):
        #Selection 1
        selection_1_inputs = Input(shape=(2*output_dim,))
        selection_1 = Dense(1, activation='linear')(selection_1_inputs)
        weighs = []
        for k in range(0,2):
            for i in range(0,output_dim):
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
    final_model = Model(inputs=inputs, outputs=final_output)
    final_model.compile(optimizer='sgd', loss='mse')
    
    net1.q_vals = final_model
    net1.dumpTo()
    net1.load()
    print(net1.q_vals.summary())
    return net1
            
