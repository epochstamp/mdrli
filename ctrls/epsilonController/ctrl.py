import numpy as np
import joblib
import os
from ctrls.controller import Controller

class EpsilonController(Controller):
    """ A controller that modifies the probability "epsilon" of taking a random action periodically.
    
    Parameters
    ----------
    initial_e : float
        Start epsilon
    e_decays : int
        How many updates are necessary for epsilon to reach eMin
    e_min : float
        End epsilon
    evaluate_on : str
        After what type of event epsilon shoud be updated periodically. Possible values: 'action', 'episode', 'epoch'.
    periodicity : int
        How many [evaluateOn] are necessary before an update of epsilon occurs
    reset_every : str
        After what type of event epsilon should be reset to its initial value. Possible values: 
        'none', 'episode', 'epoch'.
    """

    def __init__(self, initial_e=1., e_decays=10000, e_min=0.1, evaluate_on='action', periodicity=1, reset_every='none'):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        periodicity = int(periodicity)
        initial_e = float(initial_e)
        e_decays = int(e_decays)
        e_min = float(e_min)
        self._count = 0
        self._init_e = initial_e
        self._e = initial_e
        self._e_min = e_min
        self._e_decay = (initial_e - e_min) / e_decays
        self._periodicity = periodicity

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

        self._reset_on_episode = 'episode' == reset_every
        self._reset_on_epoch = 'epoch' == reset_every

    def onStart(self, agent):
        if (self._active == False):
            return

        self._reset(agent)

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return

        if self._reset_on_episode:
           self. _reset(agent)
        elif self._on_episode:
            self._update(agent)

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._reset_on_epoch:
            self._reset(agent)
        elif self._on_epoch:
            self._update(agent)

    def onActionChosen(self, agent, action):
        if (self._active == False):
            return

        if self._on_action:
            self._update(agent)

    def _reset(self, agent):
        self._count = 0
        agent._train_policy.setAttribute("epsilon", self._init_e)
        self._e = self._init_e

    def _update(self, agent):
        self._count += 1
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            agent._train_policy.setAttribute("epsilon",self._e)
            self._e = max(self._e - self._e_decay, self._e_min)
