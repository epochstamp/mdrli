import sys
import logging
import numpy as np
import argparse
sys.path.insert(0, 'utils/')
from utils import parse_conf
from os import listdir
from os.path import isfile, join, isdir


class RunInterface(object):
       
    def __init__(self):
       self.initialize()

    def initialize(self):
       self.description=""
       self.lst_common=[]

    def common_args(self,p):
       d = parse_conf("cfgs/run/common")
       d = {k: v for k, v in d.items() if k in self.lst_common}
       for k,v in d.items():
           kwargs = v
           try:
                if kwargs["type"] == "int":
                    kwargs["type"] = int
                elif d["type"] == "float":
                    kwargs["type"] = float
           except:
                pass

           p.add_argument("--" + self.__class__.__name__.lower() + "-" + k, **kwargs)
       
    def args(self,p):
       path = "cfgs/run/" + self.__class__.__name__.lower()
       if not isfile(path): return
       d = parse_conf(path)
       for k,v in d.items():
           kwargs = v
           try:
                if kwargs["type"] == "int":
                    kwargs["type"] = int
                elif kwargs["type"] == "float":
                    kwargs["type"] = float
           except:
                pass    
           p.add_argument("--" + self.__class__.__name__.lower() + "-" + k, **kwargs)
       
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
       for k,v in vars(self.params).items():
           attr = getattr(self.params,k)
           if isinstance(attr, str) and attr.lower() == "none":
                setattr(self.params,k,None) 
       self.parser = p
       
    def print_help(self):
        self.parser.print_help()
       
    def run(self):
        raise NotImplementedError()
                
                
           
        

    
    
