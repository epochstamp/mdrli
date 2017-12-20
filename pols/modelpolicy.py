import numpy as np
from pols.policy import Policy

class ModelPolicy(Policy):

    def __init__(self, n_actions,random_state):
        Policy.__init__(self, n_actions, random_state)


    def action(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        raise NotImplementedError()

    def setAttribute(self, attr, value):
        if attr == "model":
             self._model = value

    def getAttribute(self, attr):
        if attr == "model":
             return _model
        return None
