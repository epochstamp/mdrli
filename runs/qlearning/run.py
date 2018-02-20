import sys
import logging
import random
import numpy as np
from joblib import dump,load
import json
from random import randint
import hashlib
import run
sys.path.insert(0, 'data/')
sys.path.insert(0, 'conf_env/')
from utils import get_mod_object, md5_file,parse_conf, load_dump, get_mod_class, dump_dump
from run_interface import RunInterface
import argparse
from data.dataset import DataSet
from shutil import copyfile
from joblib import load,dump
from deer.agent import NeuralAgent
import deer.experiment.base_controllers as bc
from pols.greedyPolicy.pol import GreedyPolicy
from copy import deepcopy
from agent import NeuralAgent
import os
from pprint import pprint

class Qlearning(RunInterface):

    def initialize(self):
       self.description="Q-learning with a Q-network."
       self.lst_common=["out-prefix","pol-conf-file","pol-module","rng","env-conf-file","env-module","database","pol-model","max-size-episode"]

    
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
        env_params["rng"] = rng
        pol_params = parse_conf(conf_pol_dir)
        ctrl_neural_nets_params = parse_conf(conf_ctrl_neural_nets_dir)
        backend_nnet_params = parse_conf(conf_backend_nnet_dir)
        env = get_mod_object("envs",self.params.env_module,"env",rng, **env_params)
        pol = get_mod_object("pols",self.params.pol_module,"pol",env.nActions(),rng, **pol_params)
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
                dataset = load(data)
   

        if self.params.pol_model is None:
                backend_nnet_params["input_dimensions"] = env.inputDimensions()
                backend_nnet_params["n_actions"] = env.nActions()
                backend_nnet_params["random_state"] = rng
                neural_net = get_mod_object("neural_nets", self.params.backend_nnet,"neural_net", **backend_nnet_params)
                ctrl_neural_nets_params["random_state"] = rng
                ctrl_neural_nets_params["neural_network"] = neural_net
                ctrl_neural_net = get_mod_object("ctrl_neural_nets", self.params.qnetw_module, "ctrl_neural_net", env, **ctrl_neural_nets_params)
        else:
                try:
                        ctrl_neural_net = load(self.params.pol_model)
                        ctrl_neural_net.load()
                except:
                        raise AttributeError("Warm start option is corrupted - please check your QNetwork controller object has been dumped using method dumpTo")
                                
                
        pol.setAttribute("model",ctrl_neural_net)
        
        
        test_policy = GreedyPolicy(env.nActions(),rng)
        test_policy.setAttribute("model",ctrl_neural_net)
        ctrl_neural_net._batch_size = self.params.batch_size
        agent = NeuralAgent(env, ctrl_neural_net, replay_memory_size=1000000, replay_start_size=max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))), batch_size=self.params.batch_size, random_state=rng, exp_priority=0, train_policy=pol, test_policy=test_policy, only_full_history=True)
        if data is not None:
                agent._dataset = dataset
        

        

        cfg_ctrls, sections = parse_conf("cfgs/ctrl/" + self.params.acontroller_cfg,get_sections=True)
        for s in sections:
                v = cfg_ctrls[s]
                controller = get_mod_object("ctrls",s,"ctrl",**v)
                agent.attach(controller)  
        agent.run(self.params.epochs, self.params.max_size_episode)
        
        hashed = hashlib.sha1(str(pol_params).encode("utf-8") + str(env_params).encode("utf-8") + str(seed).encode("utf-8") + str(vars(self.params)).encode("utf-8")).hexdigest()
        todump = self.params.out_prefix + str(hashed)
        out = todump
        try:
                os.makedirs("dumps/ctrl_neural_nets/"+self.params.qnetw_module+"/")
        except:
                pass
        ctrl_neural_net.dumpTo("dumps/ctrl_neural_nets/"+self.params.qnetw_module+"/"+out+".dump")
        
if __name__=="__main__":
    r = Qlearning()
    r.build()
    r.run()
