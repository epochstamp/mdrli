import numpy as np
import os
from ctrls.controller import Controller
from joblib import dump,load
class NetSaveLoadController(Controller):
    """A controller that load a network at startup and save it at the end
    
    Parameters
    ----------
    input_file : the input name of the network config
    output_file : the output name of the network config
    """

    def __init__(self, input_file="in", output_file="out"):
        """Initializer.

        """
        super(self.__class__, self).__init__()
        self._input_file = input_file
        self._output_file = output_file
    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0
        if(self._input_file!="in"):
            netclass = load(self._input_file)
            netclass.load()
            agent._network = netclass
        print(agent._network.q_vals.summary())
        
    def onEnd(self, agent):
        if (self._active == False):
            return
        if(self._output_file!="out"):
            agent._network.dumpTo(self._output_file)
            agent._network.load()
        print(agent._network.q_vals.summary())
