from pols.modelpolicy import ModelPolicy
from joblib import load



class GreedyPolicy(ModelPolicy):

    def __init__(self, n_actions,random_state):
        ModelPolicy.__init__(self,n_actions,random_state)        

        




    def action(self, state):
        if not hasattr(self,"_model") or self._model is None:
                raise AttributeError("Model has not been set in this policy")
        try:
                action, V = self._model.chooseBestAction(state)
        except Exception as e:
                raise AttributeError("Model does not meet required specifications or is corrupted. Here is the error message : " + str(e))
        return action, V


    def setAttribute(self,attr,value):
        ModelPolicy.setAttribute(self,attr,value)

    def getAttribute(self,attr):
        return ModelPolicy.setAttribute(self,attr,value)
