import sys
import logging
import random
import numpy as np
from joblib import dump,load
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

class Reporter(RunInterface):

    def initialize(self):
       self.description="Model-based data generator."
       self.lst_common=["env-module","database"]

    
    def run(self):

        if self.params.database == "none":
                raise Exception("Reporter needs a database to make a report")
        dataset = load("data/" + self.params.env_module + "/" + self.params.database + ".data")
        env = get_mod_object("envs",self.params.env_module,"env",1)
        
        

        out = "data/" + self.params.env_module + "/" + self.params.database + ".csv"
        
        
        print ("Generic report (output file : "+out+")")
        states = dataset.observations()
        actions = dataset.actions()
        rewards = dataset.rewards()
        terminals = dataset.terminals()
        incr=0
        if terminals[-1]:
                incr=1
        
        terminals = terminals[:-1] if terminals[-1] else terminals
        out = open(out,"w+")
        out.write("#episodes;reward_avg;reward_var;reward_std;\n")
        rw_list = [0]
        k = 0
        for i in terminals:
                if i:
                        rw_list.append(0)
                rw_list[-1] += rewards[k]
                k += 1
        
        rw_list = np.asarray(rw_list,dtype=np.float32)
        nb_episodes = np.count_nonzero(terminals == True) + incr
        out.write(str(nb_episodes) + ";" + str(rw_list.mean()) + ";" + str(rw_list.var()) + ";" + str(rw_list.std()) + "\n")
        print(nb_episodes)
        k = 1
        states_nspace = states.shape[0]
        out.write("episode "+str(k)+";state;action;reward;reward_cumul\n")
        rew_cumul = 0
        for i in range(terminals.shape[0]):
              if terminals[i]:
                  k += 1
                  out.write("episode "+str(k)+";state;action;reward;reward_cumul\n")
                  rew_cumul = 0
              state_lst = []
              for j in range(states_nspace):
                  state_lst.append(states[j][i])
              state = flatten(state_lst)
              state_str = hashlib.sha1(str(state).encode("utf-8")).hexdigest()
              action = actions[i]
              reward = rewards[i]
              rew_cumul += reward
              out.write("#;"+str(state_str)+";"+str(action)+";"+str(reward)+";"+str(rew_cumul) + "\n")
        out.close()    
        if self.params.env_report:
                env.summarizePerformance(dataset)
                
        


if __name__=="__main__":
    r = Datagen()
    r.build()
    r.run()
