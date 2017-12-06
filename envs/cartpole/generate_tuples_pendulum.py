""" Pendulum environment launcher.
Same principles as run_toy_env. See the wiki for more details.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np

import deer.experiment.base_controllers as bc
from my_parser import process_args
from deer.agent import NeuralAgent
from deer.q_networks.q_net_theano import MyQNetwork
from pendulum_env import MyEnv as pendulum_env
from joblib import load,dump

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 1000
    EPOCHS = 200
    STEPS_PER_TEST = 1000
    PERIOD_BTW_SUMMARY_PERFS = 10

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.005
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.9
    DISCOUNT_INC = .99
    DISCOUNT_MAX = 0.95
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .2
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 100
    DETERMINISTIC = False
    CONFIG_ENV=""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(12345)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = pendulum_env(rng, conf_file=parameters.config_env)

    # --- Instantiate qnetwork ---
    qnetwork = MyQNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_delta,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng)
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)

    for i in range(1000):
    	env.reset()
	obs = env.observe()
	act = rng.choice([-env.params["F"],env.params["F"]])
	rew = env.act(act)
	agent._addSample(obs,act,rew,False)
	#print [obs,act,rew]
    dump(agent._dataset,parameters.config_env + ".dataset")
	
