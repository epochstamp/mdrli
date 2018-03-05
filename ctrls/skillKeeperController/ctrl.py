import numpy as np
import joblib
import os
from ctrls.controller import Controller

class SkillKeeperController(Controller):
    """[Experimental] A controller that tries to avoid unstabilities by changing loss function on the fly. Whenever a "best" candidate model is spot, the loss function
       is changed to a weighted one to take into account mse + kl divergence of the current learning model to the candidate model. 
    
    Parameters
    ----------
    w_mse : float 
        Initial weight for mse
    w_kld : float 
        Initial weight of kld
    w_mse_decay : float 
        Weight decay for w_mse
    w_kld_decay : float 
        Weight decay for w_kld
    """

    def __init__(self, w_mse = 0.1, w_kld = 0.9, w_decays = 10000):
        super(self.__class__, self).__init__()
        self._w_mse = w_mse
        self._w_kld = w_kld
        self._w_decay_mse = 0 if w_decays == 0 else (1 - w_mse) / w_decays
        self._w_decay_kld = 0 if w_decays == 0 else (w_kld) / w_decays
        self._pair_w = []
        

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        networks = agent.getNetworks()
        diff = len(network) - len(self._pair_w)
        for i in range(len(self._pair_w)):
                #Decrease w_kld
                self._pair_w[i][1] = max(0,self._pair_w[i][1] - self._w_mse_decay)
                #Increase w_mse
                self._pair_w[i][0] = min(1,self._w_mse_decay + self._pair_w[i][0])
        for i in range(diff):
                self._pair_w.append((self._w_mse,self._w_kld))

        for i in range(len(networks)):
                network = networks[i]
                w_mse,w_kld = self._pair_w[i]
                
                def skillkeeper_loss(network,y_true,y_pred):
                        def loss(y_true,y_pred):
                                pass
                        return loss
                network._compile(skillkeeper_loss)

