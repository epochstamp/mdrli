from pols.policy import Policy
from pols.randomPolicy.pol import RandomPolicy
from pols.greedyPolicy.pol import GreedyPolicy
from utils import parse_conf

class EpsilonGreedyPolicy(Policy):

    def __init__(self, n_actions,random_state,epsilon=0.1):
        Policy.__init__(self,n_actions,random_state)
        self._epsilon = epsilon
        self._randomPolicy = RandomPolicy(n_actions,random_state)
        self._greedyPolicy = GreedyPolicy(n_actions,random_state)




    def action(self, state):
        if self.random_state.rand() < self._epsilon:
            action, V = self._randomPolicy.action(state)
        else:
            action, V = self._greedyPolicy.action(state)

        return action, V


    def setAttribute(self,attr,value):
        if (attr == "epsilon"):
                self._epsilon = float(value)
        self._randomPolicy.setAttribute(attr,value)
        self._greedyPolicy.setAttribute(attr,value)

    def getAttribute(self,attr):
        if (attr == "epsilon"):
            return self._epsilon
        r = self._randomPolicy.getAttribute(attr)
        if r is not None:
            return r
        r = self._greedyPolicy.getAttribute(attr)
        return r
