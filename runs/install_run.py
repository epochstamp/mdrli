import os
import sys
sys.path.insert(0,".")
from run_interface import RunInterface
from argparse import ArgumentParser
from utils import list_classes_module_with_parent,copy_rename

if __name__=="__main__":
        p = ArgumentParser(description="Install a run component")
        p.add_argument("-p", "--pathfile",help="Path file of your run component", required=True, dest="pathfile")
        args = p.parse_args()
        pathfile = args.pathfile
        try:
                className = list_classes_module_with_parent(pathfile,RunInterface)[0]
        except:
                print ("Error when extracting a RunInterface-based class from your module file. Please check location and ensure that there is only one classe with RunInterface inheritance")
        moduleName = className[0].lower() + className[1:] 
        root_run = os.path.dirname(os.path.realpath(__file__))
        copy_rename(pathfile,root_run + "/" + moduleName + "/run.py")
        os.makedirs(root_run+"/../cfgs/arg/"+moduleName)
        
