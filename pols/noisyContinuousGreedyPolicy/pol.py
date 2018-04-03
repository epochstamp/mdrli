from pols.modelpolicy import ModelPolicy
from joblib import load
import numpy as np


                

class NoisyContinuousGreedyPolicy(ModelPolicy):

    def __init__(self, n_actions,random_state, noise_func='zero', noise_coeff=0):
        ModelPolicy.__init__(self,n_actions,random_state)        
        self.noise_func = noise_func
        self.noise_coeff = noise_coeff
        




    def action(self, state):
        if not hasattr(self,"_model") or self._model is None:
            raise AttributeError("Model has not been set in this policy")
        try:
            action, V = self._model.chooseBestAction(state,noise_func=self.noise_func,noise_coeff=self.noise_coeff)
        except Exception as e:
            raise AttributeError("Model does not meet required specifications or is corrupted. Here is the error message : " + str(e))
        return action, V


    def setAttribute(self,attr,value):
        ModelPolicy.setAttribute(self,attr,value)
        if attr=="noise_coeff":
            self.noise_coeff = noise_coeff
        if attr="noise_func":
            self.noise_func = noise_func

    def getAttribute(self,attr):
        return ModelPolicy.setAttribute(self,attr,value)
        if attr=="noise_coeff":
            return noise_coeff
        if attr="noise_func":
            return noise_func
