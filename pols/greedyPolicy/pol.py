from pols.policy import Policy
from joblib import load



class GreedyPolicy(Policy):

    def __init__(self, n_actions,random_state,params=None):
        Policy.__init__(self,n_actions,random_state)
        self._model = None
        if params is not None:
                self._model = params.get("MODEL_DUMP",None)
        if self._model is None:
                raise Exception("Greedy policy requires a model object to work")
        
        try:
                self._model = load("models/" + self._model)
        except:
                if not(hasattr(self._model, "chooseBestAction")):
                        raise Exception("Not able to load the model directly or by file. Please check your configuration file")

        if self._model == None:
                raise Exception("Greedy policy requires a model object to work")

        if not(hasattr(self._model, "chooseBestAction") and callable(getattr(self._model, 'chooseBestAction')) and self._model.chooseBestAction.__code__.co_argcount >= 1):
                raise Exception("Your model object is expected to have a method named 'chooseBestAction' with at least one argument.")




    def action(self, state):
        action, V = self._model.chooseBestAction(state)
        return action, V


    def setAttribute(self,attr,value):
        if attr == "model":
                self._model = value

    def getAttribute(self,attr):
        if attr == "model":
                return self._model
        return None
