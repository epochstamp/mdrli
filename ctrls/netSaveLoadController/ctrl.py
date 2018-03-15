import numpy as np
import joblib
import os
from ctrls.controller import Controller

class netSaveLoadController(Controller):
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
        #agent._network.load(self._input_file)
        
    def onEnd(self, agent):
        if (self._active == False):
            return
        #agent._network.save(self._output_file)
    
        pass
