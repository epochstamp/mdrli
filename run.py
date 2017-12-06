import sys
import logging
import numpy as np
import argparse
sys.path.insert(0, 'utils/')
from utils import get_mod_object
from run_interface import RunInterface

class Run(RunInterface):
       
    def initialize(self):
       self.description="Executes sequentially a list of routines with their options indicated by command-line input"
       self.lst_common=[]

                
       
    def run(self):
        runners = []
        for arg in vars(self.params):
            flat_runs = [item for sublist in getattr(self.params, arg) for item in sublist]
            for r in flat_runs:
                runner = get_mod_object("runs",r,"run") 
                runner.build()
                runners.append(runner)
            for r in runners:
                r.run()
                
                
           
        

if __name__ == "__main__":

    main_run = Run()
    
    main_run.build()
    
    main_run.run()
    
    
