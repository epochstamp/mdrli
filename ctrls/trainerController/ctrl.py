import numpy as np
import joblib
import os
from ctrls.controller import Controller
class TrainerController(Controller):
    """A controller that makes the agent train on its current database periodically.

    Parameters
    ----------
    evaluate_on : str
        After what type of event the agent shoud be trained periodically. Possible values: 
        'action', 'episode', 'epoch'. The first training will occur after the first occurence of [evaluateOn].
    periodicity : int
        How many [evaluateOn] are necessary before a training occurs
        _show_avg_Bellman_residual [bool] - Whether to show an informative message after each episode end (and after a 
        training if [evaluateOn] is 'episode') about the average bellman residual of this episode
    show_episode_avg_V_value : bool
        Whether to show an informative message after each episode end (and after a 
        training if [evaluateOn] is 'episode') about the average V value of this episode
    """
    def __init__(self, evaluate_on='action', periodicity=1, show_episode_avg_V_value=True, show_avg_Bellman_residual=True, training_repeat=1):
        """Initializer.
        """
        super(self.__class__, self).__init__()
        show_episode_avg_V_value = bool(show_episode_avg_V_value)
        show_avg_Bellman_residual = bool(show_avg_Bellman_residual)
        periodicity = int(periodicity)
        
        self._count = 0
        self._periodicity = periodicity
        self._show_avg_Bellman_residual = show_avg_Bellman_residual
        self._show_episode_avg_V_value = show_episode_avg_V_value
        self._training_repeat = training_repeat
        self._on_action = "action" == evaluate_on
        self._on_episode = "episode" == evaluate_on
        self._on_epoch = "epoch" == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True
        
        
    def onStart(self, agent):
        if (self._active == False):
            return
        
        self._count = 0

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return
        
        if self._on_episode:
            self._update(agent)

        if self._show_avg_Bellman_residual: print("Average (on the epoch) training loss: {}".format(agent.avgBellmanResidual()))
        if self._show_episode_avg_V_value: print("Episode average V value: {}".format(agent.avgEpisodeVValue())) # (on non-random action time-steps)

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._on_epoch:
            self._update(agent)

    def onActionTaken(self, agent):
        if (self._active == False):
            return

        if self._on_action:
            self._update(agent)

    def _update(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            for i in range(self._training_repeat) : agent.train()
        self._count += 1
