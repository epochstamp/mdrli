from pols.policy import Policy
from pols.randomPolicy.pol import RandomPolicy
from pols.greedyPolicy.pol import GreedyPolicy
import numpy as np
class EpsilonGreedyLowFreqPolicy(Policy):

    def __init__(self, n_actions,random_state,epsilon=0.1,rep_action_max_num=50):
        Policy.__init__(self,n_actions,random_state)
        self._epsilon = epsilon
        self._randomPolicy = RandomPolicy(n_actions,random_state)
        self._greedyPolicy = GreedyPolicy(n_actions,random_state)
        self._rep_counter=0
        self._rep_action_max_num=rep_action_max_num
        self._rep_action=0
        self._rep_V=0




    def action(self, state):
        if(self._rep_counter==0):
            if self.random_state.rand() <= self._epsilon:
                action, V = self._randomPolicy.action(state)
                self._rep_action = action
                self._rep_V = V
                self._rep_counter = self.random_state.randint(0, self._rep_action_max_num)
            else:
                action, V = self._greedyPolicy.action(state)
        else:
            action = self._rep_action
            V = self._rep_V
            self._rep_counter = self._rep_counter - 1
        return action, V


    def setAttribute(self,attr,value):
        if (attr == "epsilon"):
                self._epsilon = float(value)
        if (attr == "rep_action_max_num"):
            self._rep_action_max_num = int(value)
        self._randomPolicy.setAttribute(attr,value)
        self._greedyPolicy.setAttribute(attr,value)

    def getAttribute(self,attr):
        if (attr == "epsilon"):
            return self._epsilon
        if (attr == "rep_action_max_num"):
            return self._rep_action_max_num
        r = self._randomPolicy.getAttribute(attr)
        if r is not None:
            return r
        r = self._greedyPolicy.getAttribute(attr)
        return r
