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

class Tester(RunInterface):

    def initialize(self):
       self.description="Print generic and env-based reports of a policy applied in an environment"
       self.lst_common=[]

    
    def run(self):
        pass

if __name__=="__main__":
    r = Tester()
    r.build()
    r.run()
