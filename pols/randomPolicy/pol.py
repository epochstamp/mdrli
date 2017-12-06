from pols.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, n_actions,random_state,params=None):
        Policy.__init__(self,n_actions,random_state)
        




    def action(self, state):
        if ( isinstance(self.n_actions,int)):
            # Discrete set of actions [0,nactions[
            action = self.random_state.randint(0, self.n_actions)
        else:
            # Continuous set of actions
            action=[]
            for a in self.n_actions:
                action.append( self.random_state.uniform(a[0],a[1]) )
            action=np.array(action)

        V = 0
        return action, V
