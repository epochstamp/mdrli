import sys
import logging
import random
import numpy as np
from joblib import dump, load
import json
from random import randint
import hashlib
import run
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'data/')
sys.path.insert(0, 'conf_env/')
from utils import get_mod_object, md5_file,parse_conf,flatten
from run_interface import RunInterface
import argparse
from data.dataset import DataSet
from shutil import copyfile
from copy import deepcopy
from multiprocessing import Pool
from time import time

def parallel_episode(args):
        max_size_episode = args[0]
        env = args[1]
        pol = args[2]
        pol_model = args[3]
        env.reset()
        l = []
        if pol_model is not None:
                ctrl_neural_net = load(pol_model)
                ctrl_neural_net.load()
                pol.setAttribute("model",ctrl_neural_net)
        for j in range(int(max_size_episode)):
                obs = env.observe()
                act,_ = pol.action(obs)
                r = env.act(act)
                l.append((obs,act,r,j == int(max_size_episode) - 1 or env.inTerminalState(),1))
                if env.inTerminalState():
                    break
        return l
        

class Datagen(RunInterface):

    def initialize(self):
       self.description="Model-based data generator."
       self.lst_common=["out-prefix","max-size-episode","pol-conf-file","pol-module","rng","n-episodes","env-conf-file","env-module","pol-model","threads"]

    
    def run(self):

    
        if self.params.rng == -1:
                seed = random.randrange(2**32 - 1)
        else:
                seed = int(self.params.rng)
        
        rng = np.random.RandomState(seed)

        conf_env_file = self.params.env_conf_file
        conf_env_dir = "cfgs/env/" + self.params.env_module + "/" + conf_env_file
        conf_pol_file = self.params.pol_conf_file
        conf_pol_dir = "cfgs/pol/" + self.params.pol_module + "/" + conf_pol_file
        env_params = parse_conf(conf_env_dir)
        pol_params = parse_conf(conf_pol_dir)
        env = get_mod_object("envs",self.params.env_module,"env",rng, **env_params)
        pol = get_mod_object("pols",self.params.pol_module,"pol",env.nActions(),rng, **pol_params)
        dataset = DataSet(env)
        hashed = hashlib.sha1(str(pol_params).encode("utf-8") + str(env_params).encode("utf-8") + str(seed).encode("utf-8") + str(vars(self.params)).encode("utf-8")).hexdigest()
        data = self.params.out_prefix + str(hashed)
        out = "data/" + self.params.env_module + "/" + data
    
        
    
        currTime = time()
        if self.params.threads == 1:
                if self.params.pol_model is not None:
                        ctrl_neural_net = load(self.params.pol_model)
                        ctrl_neural_net.load()
                        pol.setAttribute("model",ctrl_neural_net)
                for i in range(int(self.params.n_episodes)):
                    env.reset()
                    for j in range(int(self.params.max_size_episode)):
                        obs = env.observe()
                        act,_ = pol.action(obs)
                        r = env.act(act)
                        dataset.addSample(obs, act, r, j == int(self.params.max_size_episode) - 1 or env.inTerminalState(), 1)
                        if env.inTerminalState():
                            break
        else:
                p = Pool(self.params.threads)
                lst = p.map(parallel_episode,[[self.params.max_size_episode, deepcopy(env), deepcopy(pol), self.params.pol_model]]*self.params.n_episodes)
                lst = flatten(lst,ltypes=(list))
                for o,a,r,t,p in lst:
                      dataset.addSample(o,a,r,t,p)
        print ("Execution time : " + str(time() - currTime))
        
        dump(dataset,out + ".data")
        f = open("data/" + self.params.env_module + "/last","w+")
        f.write(data)
        f.close()
        



if __name__=="__main__":
    r = Datagen()
    r.build()
    r.run()
