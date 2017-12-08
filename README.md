# Montefiore Deep Reinforcement Learning Interface (mdrli)
## Overview

MDRLI is an interface built on top of a fork based on [DeeR][(https://github.com/VinF/deer/)]. It is a plugin based-system for which you can use (already existing or provided by your own) environments and deep reinforcement learning techniques (including transfer) together very easily from data generation to learning and test.

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
 
 Everything starts by the following bash command.
 
 ```bash
python run.py --runs-run `my_run` [`my_runs` ...]
```

Any argument to be provided to `my_run` always carries the prefix `--my-run-`.

## Common Arguments

  Available common command-line arguments are listed below.
  - `env-module` : Module from which environment class (inherited from [Environment](https://github.com/VinF/deer/blob/master/deer/base_classes/Environment.py) is imported. Name of the class is required to match `env-module` with first letter capitalized.
  - `env-conf-file` : Configuration file provided to environment object imported from `env-module`. See [ConfigObj](http://configobj.readthedocs.io/en/latest/configobj.html) documentation for full specs
  - `max-size-episode` : Maximum size of an episode (related to reinforcement learning)
  - `n-episodes` : Number of episodes to play (related to reinforcement learning)
  - `out-prefix` : Prefix of output file encoded based on command line arguments.
  - `rng` : Seed for random number generator. If seed = -1, a random seed is provided.
  - `pol-module` : Module from which a policy class (inherited from [Policy](https://github.com/epochstamp/mdrli/tree/master/pols) will be imported. Name of the class is required to match `pol-module` with first letter capitalized.
  
## Runs

Runs are located [here](https://github.com/epochstamp/mdrli/tree/master/runs). See `README.md`'s in each run folder for documentation.
  
  
     
     
## How to contribute

    This interface is plugin-based. Each `README.md`, if applies, provides instructions to grow the interface. Main acceptance criterion is genericity (i.e., does not apply only in a particular context). 

     
    
 
