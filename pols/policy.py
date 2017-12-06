import numpy as np

class Policy(object):

    def __init__(self, n_actions,random_state,params=None):
        self.n_actions = n_actions
        self.random_state = random_state


    def action(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        raise NotImplementedError()

    def setAttribute(self, attr, value):
        pass

    def getAttribute(self, attr, value):
        return None
