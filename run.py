import sys
import logging
import random
import numpy as np
from joblib import dump,load
import json
from random import randint
import hashlib
from utils.utils import get_mod_object, md5_file,parse_conf, load_dump, get_mod_class, dump_dump, flatten, write_conf, erase_dict_from_keyword_list, revalidate_dict_from_conf_module
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
from os.path import isfile
import traceback
from os import remove
def custom_bool(arg):
        return arg == "True"

class Run(object):
       
    def __init__(self):
       self.initialize()

    def initialize(self):
       self.description="Main script."

       
    def args(self,p):
       path = "cfgs/args"
       if not isfile(path): return
       d = parse_conf(path)
       for k,v in d.items():
           kwargs = v
           try:
                if kwargs["type"] == "int":
                    kwargs["type"] = int
                elif kwargs["type"] == "float":
                    kwargs["type"] = float
                elif kwargs["type"] == "bool":
                    kwargs["type"] = custom_bool
           except:
                pass   
           p.add_argument("--" + k, **kwargs)
       
    def build(self):
       self.parser = argparse.ArgumentParser(description=self.description)
       try:
           self.args(self.parser)
       except argparse.ArgumentError as ae:
           print("Check cfgs/args file, it seems there is an issue there.")
           exit(-1)

       
       try:
           args = self.parser.parse_args()
       except Exception as e:
           self.parser.error(message=e)
       
       self.params = args
       for k,v in vars(self.params).items():
           attr = getattr(self.params,k)
           if isinstance(attr, str) and attr.lower() == "none":
                setattr(self.params,k,None) 
       
    def print_help(self):
        self.parser.print_help()
       
    def run(self):
        if self.params.rng == -1:
                seed = random.randrange(2**32 - 1)
        else:
                seed = int(self.params.rng)
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
    
        conf_env_dir = "cfgs/env/" + self.params.env_module + "/" + self.params.env_conf_file
        conf_ctrl_neural_nets_dir = "cfgs/ctrl_nnet/" + self.params.qnetw_module + "/" + self.params.ctrl_neural_nets_conf_file
        conf_backend_nnet_dir = "cfgs/backend_nnet/" + self.params.backend_nnet + "/" + self.params.backend_nnet_conf_file
        env_params = parse_conf(conf_env_dir)
        env_params["rng"] = rng
        ctrl_neural_nets_params = parse_conf(conf_ctrl_neural_nets_dir)
        backend_nnet_params = parse_conf(conf_backend_nnet_dir)
        env = get_mod_object("envs",self.params.env_module,"env",(rng,), env_params,mode=1)

        pol_train = get_mod_class("pols",self.params.pol_train_module, "pol")
        self.params.pol_train_args = flatten(self.params.pol_train_args) if self.params.pol_train_args is not None else [] 
        pol_train_args = parse_conf("cfgs/pol/" + self.params.pol_train_module + "/" + self.params.pol_train_args[0]) if len(self.params.pol_train_args) > 0 and isfile("cfgs/pol/" + self.params.pol_train_module + "/" + self.params.pol_train_args[0]) else parse_conf("cfgs/pol/" + self.params.pol_train_module + "/default")
        pol_train_args_2 = erase_dict_from_keyword_list(pol_train_args, self.params.pol_train_args)
        pol_train_args = revalidate_dict_from_conf_module(pol_train_args_2, "pol", self.params.pol_train_module)


        neural_net = get_mod_class("neural_nets", self.params.backend_nnet,"neural_net")
        ctrl_neural_nets_params["neural_network_kwargs"] = backend_nnet_params
        ctrl_neural_nets_params["batch_size"] = self.params.batch_size
        ctrl_neural_net = get_mod_object("ctrl_neural_nets", self.params.qnetw_module, "ctrl_neural_net", (env,),ctrl_neural_nets_params, mode=0)
        
        agent = NeuralAgent([env], [ctrl_neural_net], replay_memory_size=self.params.replay_memory_size, replay_start_size=None, batch_size=self.params.batch_size, random_state=rng, exp_priority=self.params.exp_priority, train_policy=pol_train,train_policy_kwargs=pol_train_args, only_full_history=self.params.only_full_history)
       


        for tc in self.params.controllers:
                len_tc = len(tc)                
                s = tc[0]
                redo_conf = False
                if len_tc >= 2:
                    
                    #Test if sc is a config file or an argument to override
                    if '=' not in tc[1]:
                        #This is a config file
                        conf_ctrl = parse_conf("cfgs/ctrl/" + s + "/" + tc[1])
                    else:
                        conf_ctrl = parse_conf("cfgs/ctrl/" + s + "/default")
                        sc = tc[1].split("=")
                        if sc[0] in conf_ctrl.keys():
                            conf_ctrl[sc[0]] = sc[1]
                            redo_conf = True
                        else:
                            print ("Warning : parameter " + str(sc[0]) + " is not included in config specs for the controller " + s)

                    if len_tc > 2:
                        remainder = tc[2:]
                        for a in remainder:
                             sc = a.split("=")
                             if len(sc) != 2:
                                 print ("Warning : arg " + a + " for controller parametrization is ill formed. It needs to be in the form key=value.") 
                             else:
                                 redo_conf = True
                                 if sc[0] in conf_ctrl.keys():
                                    conf_ctrl[sc[0]] = sc[1]
                                 else:
                                    print ("Warning : parameter " + str(sc[0]) + " is not included in config specs for the controller " + s)
                    #Create a temporary config file with the erased parameter and go through parse_conf again
                    if redo_conf:
                        write_conf(conf_ctrl, "cfgs/ctrl/" + s + "/temp")
                        conf_ctrl = parse_conf("cfgs/ctrl/" + s + "/temp")
                        os.remove("cfgs/ctrl/" + s + "/temp") 
                    
                else:
                    conf_ctrl = parse_conf("cfgs/ctrl/" + s + "/default")
                controller = get_mod_object("ctrls",s,"ctrl",tuple(),conf_ctrl,mode=0)
                
                agent.attach(controller)
        agent.run(self.params.epochs, self.params.max_size_episode)
                
                
           
        

if __name__ == "__main__":

    main_run = Run()
    main_run.build()
    try:
        main_run.run()
    except Exception as e:
        print(e)
        print("Traceback below")
        traceback.print_exc()
