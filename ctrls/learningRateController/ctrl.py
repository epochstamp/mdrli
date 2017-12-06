import numpy as np
import joblib
import os
from ctrls.controller import Controller

class LearningRateController(Controller):
    """A controller that modifies the learning rate periodically upon epochs end.
    
    Parameters
    ----------
    initial_learning_rate : float
        The learning rate upon agent start
    learning_rate_decay : float
        The factor by which the previous learning rate is multiplied every [periodicity] epochs.
    periodicity : int
        How many epochs are necessary before an update of the learning rate occurs
    """

    def __init__(self, initial_learning_rate=0.005, learning_rate_decay=1., periodicity=1):
        """Initializer.

        """
        super(self.__class__, self).__init__()
        initial_learning_rate = float(initial_learning_rate)
        learning_rate_decay = float(learning_rate_decay)
        periodicity = int(periodicity)
        self._epoch_count = 0
        self._init_lr = initial_learning_rate
        self._lr = initial_learning_rate
        self._lr_decay = learning_rate_decay
        self._periodicity = periodicity

    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0
        agent._network.setLearningRate(self._init_lr)
        self._lr = self._init_lr * self._lr_decay

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        self._epoch_count += 1
        if self._periodicity <= 1 or self._epoch_count % self._periodicity == 0:
            agent._network.setLearningRate(self._lr)
            self._lr *= self._lr_decay
