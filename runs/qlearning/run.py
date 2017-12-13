import sys
import logging
import random
import numpy as np
from joblib import dump
import json
from random import randint
import hashlib
import run
sys.path.insert(0, 'data/')
sys.path.insert(0, 'conf_env/')
from utils import get_mod_object, md5_file,parse_conf, load_dump, get_mod_class, dump_dump
from run_interface import RunInterface
import argparse
from deer.agent import DataSet
from shutil import copyfile
from joblib import load,dump
from deer.agent import NeuralAgent
import deer.experiment.base_controllers as bc
from pols.greedyPolicy.pol import GreedyPolicy
from copy import deepcopy
from agent import NeuralAgent
import os

class Qlearning(RunInterface):

    def initialize(self):
       self.description="Q-learning with a Q-network."
       self.lst_common=["out-prefix","pol-conf-file","pol-module","rng","env-conf-file","env-module","database"]

    
    def run(self):

        if self.params.rng == -1:
                seed = random.randrange(2**32 - 1)
        else:
                seed = int(self.params.rng)
        rng = np.random.RandomState(seed)
    
        conf_env_dir = "cfgs/env/" + self.params.env_module + "/" + self.params.env_conf_file
        conf_pol_dir = "cfgs/pol/" + self.params.pol_module + "/" + self.params.pol_conf_file
        conf_ctrl_neural_nets_dir = "cfgs/ctrl_nnet/" + self.params.qnetw_module + "/" + self.params.ctrl_neural_nets_conf_file
        conf_backend_nnet_dir = "cfgs/backend_nnet/" + self.params.backend_nnet + "/" + self.params.backend_nnet_conf_file
        env_params = parse_conf(conf_env_dir)
        pol_params = parse_conf(conf_pol_dir)
        ctrl_neural_nets_params = parse_conf(conf_ctrl_neural_nets_dir)
        backend_nnet_params = parse_conf(conf_backend_nnet_dir)
        env = get_mod_object("envs",self.params.env_module,"env",rng, env_params)
        pol = get_mod_object("pols",self.params.pol_module,"pol",rng, pol_params)
        env.reset()
        data = self.params.database
        if data == "none":
                data = None
        elif data == "last":
                try:
                        data = open("data/" + self.params.env_module + "/last").read()
                except: 
                        data = None
        if data is not None:
                dataset = load("data/" + self.params.env_module + "/" + data + ".data")

        
        backend_nnet_params["input_dimensions"] = env.inputDimensions()
        backend_nnet_params["n_actions"] = env.nActions()
        backend_nnet_params["random_state"] = rng
        neural_net = get_mod_object("neural_nets", self.params.backend_nnet,"neural_net", **backend_nnet_params)
        ctrl_neural_nets_params["random_state"] = rng
        ctrl_neural_nets_params["neural_network"] = neural_net
        ctrl_neural_net = get_mod_object("ctrl_neural_nets", self.params.qnetw_module, "ctrl_neural_net", env, **ctrl_neural_nets_params)
        if self.params.warmstart != "none":
                try:
                        ctrl_neural_net.setAllParams("dumps/neural_nets_params/" + str(self.params.qnetw_module) + "/" + self.params.warmstart + ".params")
                except:
                        print ("Warning - Warm start option is corrupted and thus not taken into account")
                                
                
        
        if "model" in pol_params.keys():
                pol_params["MODEL_DUMP"]=ctrl_neural_net

        pol_2_params = dict()
        pol_2_params["MODEL_DUMP"] = ctrl_neural_net
        agent = NeuralAgent(env, ctrl_neural_net, replay_memory_size=1000000, replay_start_size=None, batch_size=self.params.batch_size, random_state=rng, exp_priority=0, train_policy=pol, test_policy=GreedyPolicy(env.nActions(),rng, pol_2_params), only_full_history=True)
        if data is not None:
                agent._dataset = dataset
        

        

        
        cfg_ctrls = parse_conf("cfgs/ctrl/" + self.params.acontroller_cfg)
        for k,v in cfg_ctrls.items():
                controller = get_mod_object("ctrls",k,"ctrl",*v)
                agent.attach(controller)
                 
        agent.run(self.params.epochs, self.params.max_steps_on_epoch)

        hashed = hashlib.sha1(str(pol_params).encode("utf-8") + str(env_params).encode("utf-8") + str(seed).encode("utf-8") + str(vars(self.params)).encode("utf-8")).hexdigest()
        todump = self.params.out_prefix + str(hashed)
        out = todump

        dump_dump("neural_nets_params", self.params.qnetw_module, out,agent._network.getAllParams()) 
        dump_dump("ctrl_neural_nets", self.params.qnetw_module, out,agent._network)  

if __name__=="__main__":
    r = Qlearning()
    r.build()
    r.run()
