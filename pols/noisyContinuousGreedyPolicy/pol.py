from pols.modelpolicy import ModelPolicy
from joblib import load
import numpy as np


class NoisyContinuousGreedyPolicy(ModelPolicy):

    def __init__(self, n_actions,random_state, noisy = True):
        ModelPolicy.__init__(self,n_actions,random_state)  
        self.noisy = noisy




    def action(self, state):
        if not hasattr(self,"_model") or self._model is None:
            raise AttributeError("Model has not been set in this policy")
        try:
            action, V = self._model.chooseBestAction(state, self.noisy)
        except Exception as e:
            raise AttributeError("Model does not meet required specifications or is corrupted. Here is the error message : " + str(e))

        #print(V)
        return action, V


    def setAttribute(self,attr,value):
        ModelPolicy.setAttribute(self,attr,value)
        if attr=="noisy":
            self.noisy = value

    def getAttribute(self,attr):
        if attr=="noisy":
            return self.noisy
        return ModelPolicy.getAttribute(self,attr)
        
