import sys
import logging
import numpy as np
import argparse
sys.path.insert(0, 'utils/')
from utils import parse_conf
from os import listdir
from os.path import isfile, join


class RunInterface(object):
       
    def __init__(self):
       self.initialize()

    def initialize(self):
       self.description=""
       self.lst_common=[]

    def common_args(self,p):
       for c in self.lst_common:
           d = parse_conf("cfgs/arg/common/"+c)
           try:
                if d["type"] == "int":
                    d["type"] = int
                elif d["type"] == "float":
                    d["type"] = float
           except:
                pass    
           p.add_argument("--" + self.__class__.__name__.lower() + "-" + c, **d)
       
    def args(self,p):
       path = "cfgs/arg/" + self.__class__.__name__.lower() + "/"
       commands = [f for f in listdir(path) if isfile(join(path, f))]
       for c in commands:
           d = parse_conf(path+c)
           try:
                if d["type"] == "int":
                    d["type"] = int
                elif d["type"] == "float":
                    d["type"] = float
           except:
                pass    
           p.add_argument("--" + self.__class__.__name__.lower() + "-" + c, **d)
       
    def build(self):
       p = argparse.ArgumentParser(description=self.description)
       self.common_args(p)
       try:
           self.args(p)
       except argparse.ArgumentError as ae:
           print("Conflicting option with common args. Please check first that your arg is not a common one. See common_args folder at the root.")
           exit(-1)

       
       try:
           args, unknown = p.parse_known_args()
       except:
           args = None
       self.params = args
       self.parser = p
       
    def print_help(self):
        self.parser.print_help()
       
    def run(self):
        raise NotImplementedError()
                
                
           
        

    
    
