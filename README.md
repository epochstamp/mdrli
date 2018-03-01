# LegoRL
## Overview

LegoRL is an interface built on top of a fork based on [DeeR](https://github.com/VinF/deer/). It is a plugin based-system for which you can use (already existing or provided by your own) environments and deep reinforcement learning techniques (including transfer) together very easily from data generation to learning and test.

## Dependencies

This interface has been tested in Python >= 3.6 but should also work with older versions.

You will need the following dependencies : 
 - DeeR
 - NumPy >= 1.10
 - joblib >= 0.9
 - Theano >= 0.8
 - TensorFlow >= 0.9
 - Matplotlib >= 1.1.1 for testing purpose
 - Keras
 - ConfigObj >= 5.0
 
 ## How To Install
 
 For the moment, just download this package. No need to install, you can directly work by the root of the package folder.
 
 ## How To Start
 
 Run the command `python run.py` to get extensive help about arguments. We illustrate here how to run a complete deep RL workflow in environment Cartpole.

 Let us consider the following command : 

  
     
     
## How to contribute

    You can : 
 
       - Add new environments in folder envs, following [Environment](https://github.com/epochstamp/mdrli/envs/env.py) abstract class specs in this [folder](https://github.com/epochstamp/mdrli/envs/)
       - Add new controllers in folder ctrls, following [Controller](https://github.com/epochstamp/mdrli/ctrls/controller.py) abstract class specs in this [folder](https://github.com/epochstamp/mdrli/ctrls/)
       - Add new neural networks controllers, following [QNetwork](https://github.com/epochstamp/mdrli/ctrl_neural_nets/ctrl_neural_net.py) abstract class specs in this [folder](https://github.com/epochstamp/mdrli/ctrl_neural_nets/),
       - Add new neural network backend in this [folder](https://github.com/epochstamp/mdrli/neural_nets/), as long as you implement the method _buildDQN and with which neural network controller is it compatible.
    
    Approval through pull request is subject to genericity, i.e. your contribution can be used across RL domains.
