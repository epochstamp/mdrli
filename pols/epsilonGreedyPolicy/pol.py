from pols.policy import Policy
from pols.randomPolicy.pol import RandomPolicy
from pols.greedyPolicy.pol import GreedyPolicy

class EpsilonGreedyPolicy(Policy):

    def __init__(self, n_actions,random_state,epsilon=0.1,greedy_exclude_prob=None,e_dist=0.1):
        Policy.__init__(self,n_actions,random_state)
        self._epsilon = epsilon
        self._e_dist = e_dist
        self._greedy_prob_select = 1-greedy_exclude_prob if greedy_exclude_prob is not None else None
        self._randomPolicy = RandomPolicy(n_actions,random_state)
        self._greedyPolicy = GreedyPolicy(n_actions,random_state)




    def action(self, state):
        if self.random_state.rand() <= self._epsilon:                 
            if self._greedy_prob_select is None:
                action, V = self._randomPolicy.action(state)
            else:
                action, V = self._greedyPolicy.action(state)
                if ( isinstance(self.n_actions,int)):
                    ws = [(1-self._greedy_prob_select)/float(self.n_actions-1)]*self.n_actions
                    ws[action] = self._greedy_prob_select
                    action = self.random_state.choice(range(0, self.n_actions), p=ws)
                    
                else:
                    # Continuous set of actions
                    actions=[]
                    for a in self.n_actions:
                        region_inside = (action-self._e_dist,action_self._e_dist)
                        region_outside_1=(a[0],max(a[0],min(a[1],action-self._e_dist)))
                        region_outside_2=(max(a[0],min(a[1],action+self._e_dist)),a[1])
                        interval = self.random_state.choice([region_inside,region_outside_1,region_outside_2],[self._greedy_prob_select,(1-self._greedy_prob_select)/2.0,(1-self._greedy_prob_select)/2.0])
                        actions.append(self.random_state.uniform(interval))
                    action=np.array(actions)
                V = 0

        else:
            action, V = self._greedyPolicy.action(state)

        return action, V


    def setAttribute(self,attr,value):
        if (attr == "epsilon"):
            self._epsilon = float(value)
        elif (attr == "e_dist"):
            self._e_dist = float(value)
        elif (attr=="greedy_exclude_prob"):
            greedy_exclude_prob = value
            self._greedy_prob_select = 1-greedy_exclude_prob if greedy_exclude_prob is not None else None
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
