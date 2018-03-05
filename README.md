# LegoRL
## Overview

LegoRL is a reinforcement learning plugin-based agent built on top of a fork based on [DeeR](https://github.com/VinF/deer/). A LegoRL agent may be mono or multi-task (which make easier implementation of transfer learning techniques) and use a Q-Network to learn. Furthermore, the LegoRL agent work with third-party classes (that are named "controllers") to perform exploration/exploitation routines (e.g. controlling learning rate, discount factor, €-exploration, and more...)  

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
 
 There is no setup, you can directly use the main script by running `python run.py`, which will give you an extensive help about arguments.
 
 ## How To Start
 
 The only required argument is the environment module to consider. Here, we run a minimal example with Cartpole :

 `python run.py --env-module cartpole`.

 However, while there is a Reinforcement Learning loop running with this command, only exploration is performed, and thus a neural network file with random weights is output. If you want to include training, you need to attach the appropriate controller as done below : 

 `python run.py --env-module cartpole --attach trainerController`.

 Now learning is included in the RL loop. [Default configuration](https://github.com/epochstamp/mdrli/cfgs/ctrl/trainerController/default) will be used here. Below is the complete synopsis of adding a controller : 
 `python run.py --env-module cartpole --attach trainerController [config_file | key=value] [key=value]^+` where the configuration file `config_file`, if provided, is loaded and the following arguments are used to override the values in `config_file`. Otherwise, the values are used to override those are defined in the default configuration file. Go to the [Controller folder](https://github.com/epochstamp/mdrli/ctrls/) to see which controllers you can include in the command. The controllers are executed sequentially according to the order in which they are declared in the command line. For example, when the following command is run, 

`python run.py --env-module cartpole --attach epsilonController --attach trainerController`,

the controller EpsilonController (which decreases linearly the €-exploration rate) will be executed before the trainerController.
  
    
     
## How to contribute

You can : 

  - Add new environments in folder envs, following [Environment](https://github.com/epochstamp/mdrli/envs/env.py) abstract class specs in this [folder](https://github.com/epochstamp/mdrli/envs/)
  - Add new controllers in folder ctrls, following [Controller](https://github.com/epochstamp/mdrli/ctrls/controller.py) abstract class specs in this [folder](https://github.com/epochstamp/mdrli/ctrls/)
  - Add new neural networks controllers, following [QNetwork](https://github.com/epochstamp/mdrli/ctrl_neural_nets/ctrl_neural_net.py) abstract class specs in this [folder](https://github.com/epochstamp/mdrli/ctrl_neural_nets/),
  - Add new neural network backend in this [folder](https://github.com/epochstamp/mdrli/neural_nets/), as long as you implement the method _buildDQN and with which neural network controller is it compatible.

Approval through pull request is subject to genericity, i.e. your contribution can be used across RL domains.
