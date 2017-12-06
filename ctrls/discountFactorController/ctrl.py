import numpy as np
import joblib
import os
from ctrls.controller import Controller

class DiscountFactorController(Controller):
    """A controller that modifies the q-network discount periodically.
    More informations in : Francois-Lavet Vincent et al. (2015) - How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies (http://arxiv.org/abs/1512.02011).

    Parameters
    ----------
    initial_discount_factor : float
        Start discount
    discount_factor_growth : float
        The factor by which the previous discount is multiplied every [periodicity]
        epochs.
    discount_factor_max : float
        Maximum reachable discount
    periodicity : int
        How many training epochs are necessary before an update of the discount occurs
    """
    
    def __init__(self, initial_discount_factor=0.9, discount_factor_growth=1., discount_factor_max=0.99, periodicity=1):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        periodicity = int(periodicity)
        initial_discount_factor = float(initial_discount_factor)
        discount_factor_growth = float(discount_factor_growth)
        discount_factor_max = float(discount_factor_max)
        self._epoch_count = 0
        self._init_df = initial_discount_factor
        self._df = initial_discount_factor
        self._df_growth = discount_factor_growth
        self._df_max = discount_factor_max
        self._periodicity = periodicity

    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0
        agent._network.setDiscountFactor(self._init_df)
        if (self._init_df < self._df_max):
            self._df = 1 - (1 - self._init_df) * self._df_growth
        else:
            self._df = self._init_df

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        self._epoch_count += 1
        if self._periodicity <= 1 or self._epoch_count % self._periodicity == 0:
            if (self._df < self._df_max):
                agent._network.setDiscountFactor(self._df)
                self._df = 1 - (1 - self._df) * self._df_growth
