import numpy as np
import joblib
import os
from ctrls.controller import Controller

class TesterController(Controller):
    """A controller that performs several tests on an environment with a testing policy
    
    Parameters
    ----------
    id : int
        The identifier (>= 0) of the mode each test epoch triggered by this controller will belong to. 
        Can be used to discriminate between datasets in your Environment subclass (this is the argument that 
        will be given to your environment's reset() method when starting the test epoch).
    epoch_length : float
        The total number of transitions that will occur during a test epoch. This means that
        this epoch could feature several episodes if a terminal transition is reached before this budget is 
        exhausted.
    periodicity : int 
        How many epochs are necessary before tests are ran.
    show_score : bool
        Whether to print an informative message on stdout at the end of each test epoch, about 
        the total reward obtained in the course of the test epoch.
    number_tests : int
        How many tests the controller will perform on the environment
    """

    def __init__(self, id=0, epoch_length=500, periodicity=2, show_score=True, number_tests=1):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        periodicity = int(periodicity)
        id = int(id)
        epoch_length = int(epoch_length)
        controllers_to_disable = map(int, controllers_to_disable.split(","))
        show_score = bool(show_score)
        summarize_every = int(summarize_every)
        number_tests = int(number_tests)
        self._epoch_count = 0
        self._id = id
        self._epoch_length = epoch_length
        self._to_disable = controllers_to_disable
        self._show_score = show_score
        self._periodicity = periodicity
        self._number_tests = number_tests


    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        mod = self._epoch_count % self._periodicity
        self._epoch_count += 1
        if mod == 0:
            agent.startMode(self._id, self._epoch_length)
            agent.setControllersActive(self._to_disable, False)
        elif mod == 1:
            self._summary_counter += 1
            if self._show_score:
                score,nbr_episodes=agent.totalRewardOverLastTest()
                print("Testing score per episode (id: {}) is {} (average over {} episode(s))".format(self._id, score, nbr_episodes))
            if self._summary_periodicity > 0 and self._summary_counter % self._summary_periodicity == 0:
                agent.summarizeTestPerformance()
            agent.resumeTrainingMode()
            agent.setControllersActive(self._to_disable, True)
