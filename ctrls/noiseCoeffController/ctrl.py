import numpy as np
import joblib
import os
from ctrls.controller import Controller

def plus(a,b):
        return a+b

def mul(a,b)
        return a*b

class NoiseCoeffController(Controller):
    """ A controller that modifies the noise coefficient to apply in a continuous action periodically.
    
    Parameters
    ----------
    initial_coeff : float
        Start noise coefficient
    coeff_decay : int
        Growing delta for noise coefficient
    coeff_operator : str
        Growing operator for noise coefficient (needs to be 'plus' or 'mul')
    evaluate_on : str
        After what type of event epsilon shoud be updated periodically. Possible values: 'action', 'episode', 'epoch'.
    periodicity : int
        How many [evaluateOn] are necessary before an update of epsilon occurs
    reset_every : str
        After what type of event epsilon should be reset to its initial value. Possible values: 
        'none', 'episode', 'epoch'.
    """

    def __init__(self, init_coeff=1., coeff_decay=1, coeff_operator='plus', evaluate_on='action', periodicity=1, reset_every='none'):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        periodicity = int(periodicity)
        init_coeff = float(init_coeff)
        coeff_decay = int(coeff_decay)
        self._count = 0
        self._init_coeff = initial_coeff
        self._coeff = init_coeff
        self._coeff_decay = coeff_decay
        self._periodicity = periodicity
        self._coeff_grower = plus if coeff_operator == 'plus' else (mul if coeff_operator == 'mul' else 'plus')

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
        agent._train_policy.setAttribute("noise_coeff", self._init_coeff)
        self._coeff = self._init_coeff

    def _update(self, agent):
        self._count += 1
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            self._coeff = self._coeff_grower(self._coeff, self._coeff_decay)
            agent._train_policy.setAttribute("noise_coeff",self._coeff)
            
