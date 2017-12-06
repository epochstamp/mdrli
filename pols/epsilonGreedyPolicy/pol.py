from pols.policy import Policy
from pols.policy.randomPolicy.pol import randomPolicy
from utils.utils import parse_conf

class EpsilonGreedyPolicy(Policy):

    def __init__(self, n_actions,random_state,params=None):
        Policy.__init__(self,n_actions,random_state)
        self._epsilon = 0.1
        if params is not None:
                self._epsilon = params.get("EPSILON", self._epsilon)
        conf_randomPolicy = params.get("RANDOM_POLICY_CONF","DEFAULT")
        conf_greedyPolicy = params.get("GREEDY_POLICY_CONF","DEFAULT")
        self._randomPolicy = randomPolicy(self,n_actions,random_state,parse_conf("confs/conf_pol/randomPolicy/"+conf_randomPolicy))
        self._greedyPolicy = greedyPolicy(self,n_actions,random_state,parse_conf("confs/conf_pol/greedyPolicy/"+conf_greedyPolicy))
        




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

    def getAttribute(self,attr,value):
        if (attr == "epsilon"):
                return self._epsilon
        r = self._randomPolicy.getAttribute(attr)
        if r is not None:
                return r
        r = self._greedyPolicy.getAttribute(attr)
        return r
