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
        flat_runs = [item for sublist in getattr(self.params, "runs") for item in sublist]
        display_help = self.params.man
        for r in flat_runs:
            runner = get_mod_object("runs",r,"run") 
            runner.build()
            runners.append(runner)
        for r in runners:
            if not display_help:
                try:
                    r.run()
                except AttributeError as e:
                    r.print_help()
                    raise AttributeError("Check your command-line arguments, you got the following error : " + str(e))
            else:
                r.print_help()
        
            """
            flat_runs = [item for sublist in getattr(self.params, arg) for item in sublist]
            for r in flat_runs:
                runner = get_mod_object("runs",r,"run") 
                runner.build()
                runners.append(runner)
            for r in runners:
                r.run()
            """
                
                
           
        

if __name__ == "__main__":

    main_run = Run()
    main_run.build()
    try:
        main_run.run()
    except Exception as e:
        print(e)
    
"""
runners = []
        flat_runs = []
        for arg in vars(self.params):
            if arg == "--runs-run":
                flat_runs += [item for sublist in getattr(self.params, arg) for item in sublist]
        for r in flat_runs:
            runner = get_mod_object("runs",r,"run") 
            runner.build()
            runners.append(runner)
        for r in runners:
    
        r.run()    
"""
