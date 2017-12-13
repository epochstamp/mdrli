import numpy as np
import joblib
import os
from ctrls.controller import Controller

class VerboseController(Controller):
    """A controller that print various agent information periodically:
    
    * Count of passed [evaluateOn]
    * Agent current learning rate
    * Agent current discount factor
    * Agent current epsilon

    Parameters
    ----------
    evaluate_on : str
        After what type of event the printing should occur periodically. Possible values: 
        'action', 'episode', 'epoch'. The first printing will occur after the first occurence of [evaluateOn].
    periodicity : int
        How many [evaluateOn] are necessary before a printing occurs
    """

    def __init__(self, evaluateOn=False, evaluate_on='epoch', periodicity=1):
        """Initializer.
        """
        evaluateOn = bool(evaluateOn)
        periodicity = int(periodicity)
        if evaluateOn is not False:
            raise Exception('For uniformity the attributes to be provided to the controllers respect PEP8 from deer0.3dev1 onwards. For instance, instead of "evaluateOn", you should now have "evaluate_on". Please have a look at https://github.com/VinF/deer/issues/28.')

        super(self.__class__, self).__init__()
        self._count = 0
        self._periodicity = periodicity
        self._string = evaluate_on

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_epoch = True

    def onStart(self, agent):
        if (self._active == False):
            return
        
        self._count = 0

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return
        
        if self._on_episode:
            self._print(agent)

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._on_epoch:
            self._print(agent)

    def onActionTaken(self, agent):
        if (self._active == False):
            return

        if self._on_action:
            self._print(agent)

    def _print(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            print("{} {}:".format(self._string, self._count + 1))
            print("Learning rate: {}".format(agent._network.learningRate()))
            print("Discount factor: {}".format(agent._network.discountFactor()))
            print("Database size: {}".format(agent._dataset.terminals().shape[0]))
            try:
                print("Epsilon: {}".format(agent._train_policy.epsilon()))
            except:
                pass
        self._count += 1
