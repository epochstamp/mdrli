import numpy as np
import joblib
import os
from ctrls.controller import Controller
from joblib import dump,load
from utils.fusion import fusion
class NetFusionController(Controller):
    """A controller that loads 2 networks at startup and fusion them and save the result
    
    Parameters
    ----------
    input_file1 : the input name of the first network config
    input_file2 : the input name of the second network config
    output_file : the output name of the network config
    """

    def __init__(self, input_file1="in2", input_file2="in2", output_file="out"):
        """Initializer.

        """
        super(self.__class__, self).__init__()
        self._input_file1 = input_file1
        self._input_file2 = input_file2
        self._output_file = output_file
        self._fusion_net = None
    def onStart(self, agent):
        if (self._active == False):
            return

        net1class = load(self._input_file1)
        net1class.load();
        
        net2class = load(self._input_file2)
        net2class.load();
        
        fusion_net = fusion(net1class,net2class)
        agent._network = fusion_net
