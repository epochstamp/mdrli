import numpy as np

class Policy(object):

    def __init__(self, n_actions,random_state):
        self.n_actions = n_actions
        self.random_state = random_state


    def action(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        raise NotImplementedError()
        
    def randomAction(self):
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

    def setAttribute(self, attr, value):
        pass

    def getAttribute(self, attr):
        return None
