"""
.. Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
import numpy as np

class QNetwork(object):
    """ All the Q-networks and actor-critic networks should inherit this interface.

    Parameters
    -----------
    environment : object from class Environment
        The environment linked to the Q-network
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    """
    def __init__(self, environment, batch_size):
        self._environment = environment
        self._df = 0.9
        self._lr = 0.005
        self._input_dimensions = self._environment.inputDimensions()
        self._n_actions = self._environment.nActions()
        self._batch_size = batch_size

    def train(self, states, actions, rewards, nextStates, terminals):
        """ This method performs the Bellman iteration for one batch of tuples.
        """
        raise NotImplementedError()

    def chooseBestAction(self, state,**kwargs):
        """ Get the best action for a belief state
        """        
        raise NotImplementedError()

    def qValues(self, state):
        """ Get the q value for one belief state
        """        
        raise NotImplementedError()
        
    def dumpTo(self, out_file):
        raise NotImplementedError()
        
    def load(self):
        raise NotImplementedError()

    def setLearningRate(self, lr):
        """ Setting the learning rate

        Parameters
        -----------
        lr : float
            The learning rate that has to bet set
        """
        self._lr = lr

    def setDiscountFactor(self, df):
        """ Setting the discount factor

        Parameters
        -----------
        df : float
            The discount factor that has to bet set
        """
        if df < 0. or df > 1.:
            raise AgentError("The discount factor should be in [0,1]")

        self._df = df

    def learningRate(self):
        """ Getting the learning rate
        """
        return self._lr

    def discountFactor(self):
        """ Getting the discount factor
        """
        return self._df

    def updateLossFunction(self, loss=None, args = None,kwargs = None):
        raise NotImplementedError()
        

if __name__ == "__main__":
    pass
