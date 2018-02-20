import numpy as np
import joblib
import os
from ctrls.controller import Controller

class InterleavedTestEpochController(Controller):
    """A controller that interleaves a test epoch between training epochs of the agent.
    
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
    controllers_to_disable : list of int
        A list of controllers to disable when this controller wants to start a
        test epoch. These same controllers will be reactivated after this controller has finished dealing with
        its test epoch.
    periodicity : int 
        How many epochs are necessary before a test epoch is ran (these controller's epochs
        included: "1 test epoch on [periodicity] epochs"). Minimum value: 2.
    show_score : bool
        Whether to print an informative message on stdout at the end of each test epoch, about 
        the total reward obtained in the course of the test epoch.
    summarize_every : int
        How many of this controller's test epochs are necessary before the attached agent's 
        summarizeTestPerformance() method is called. Give a value <= 0 for "never". If > 0, the first call will
        occur just after the first test epoch.
    """
    """
    def __init__(self, id=0, epoch_length=500, controllers_to_disable=[], periodicity=2, show_score=True, summarize_every=100, number_tests=10,path_files=".",prefix_file=""):
            Initializer.
            

            super(self.__class__, self).__init__()
            periodicity = int(periodicity)
            id = int(id)
            epoch_length = int(epoch_length)
            controllers_to_disable = map(int, controllers_to_disable)
            show_score = bool(show_score)
            summarize_every = int(summarize_every)
            self._epoch_count = 0
            self._id = id
            self._epoch_length = epoch_length
            self._to_disable = controllers_to_disable
            self._show_score = show_score
            if periodicity <= 2:
                self._periodicity = 2
            else:
                self._periodicity = periodicity

            try:
                os.makedirs(path_files)
            except:
                pass

            self._summary_counter = 0
            self._summary_periodicity = int(summarize_every)
            self._number_tests = int(number_tests)
            self._path_files=path_files
            self._prefix_file = prefix_file
            self._test_counter = 0

        def onStart(self, agent):
            if (self._active == False):
                return

            self._epoch_count = 0
            self._summary_counter = 0
            f = open(self._path_files + "/" + self._prefix_file + "_summary.csv", "w+")
            f.write("epoch;test;nb_episodes;r_mean;r_var\n")
            f.close()

        def onEpochEnd(self, agent):
            if (self._active == False):
                return

            mod = self._epoch_count % (self._periodicity + self._number_tests)
            

            self._epoch_count += 1
            print(mod)
            if mod == 0:
                self.score = []
                self.nbr_episodes = 0
                agent.startMode(self._id, self._epoch_length)
                agent.setControllersActive(self._to_disable, False)
            elif mod > 0 and mod <= self._number_tests - 1:
                self._summary_counter += 1
                print(self._show_score)
                if self._show_score:
                    score,nbr_episodes=agent.totalRewardOverLastTest()
                    self.score.append(score)
                    self.nbr_episodes += nbr_episodes
                    print("Testing score per episode (id: {}) is {} (average over {} episode(s))".format(self._id, score, nbr_episodes))
                if self._summary_periodicity > 0 and self._summary_counter % self._summary_periodicity == 0:
                    agent.summarizeTestPerformance(path_dump=self._path_files,prefix_file=self._prefix_file)
                if mod < self._number_tests - 1 : 
                    agent.startMode(self._id, self._epoch_length)
                elif mod == self._number_tests - 1 :
                    agent.resumeTrainingMode()
                    agent.setControllersActive(self._to_disable, True)  
                    f = open(self._path_files + "/" + self._prefix_file + "_summary.csv", "a+")
                    f.write(str(self._epoch_count) + ";" + str(self._number_tests) + ";" + str(self._epoch_count) + ";" + str(self.nbr_episodes) + ";" + str(np.mean(np.asarray(self.score)))+";"+str(np.var(np.asarray(self.score)))+"\n")
                    f.close()
            else:
                print("here")
                agent.resumeTrainingMode()
                agent.setControllersActive(self._to_disable, True)  

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
    """
    def __init__(self, id=0, epoch_length=500, controllers_to_disable=[], periodicity=2, show_score=True, summarize_every=10, number_tests=10,path_files=".",prefix_file=""):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._periodicity = max(2,int(periodicity))
        id = int(id)
        self._epoch_length = int(epoch_length)
        self._to_disable = list(map(int, controllers_to_disable))
        self._show_score = bool(show_score)
        self._epoch_count = 0
        self._id = int(id)
        self._epoch_length = int(epoch_length)

        
        self._summary_counter = 0
        self._summary_periodicity = int(summarize_every)

        self._number_tests = int(number_tests)
        self._path_files=path_files
        self._prefix_file = prefix_file

        
    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0
        self._summary_counter = 0
        try:
            os.makedirs(self._path_files)
        except:
            pass
        f = open(self._path_files + "/" + self._prefix_file + "_summary.csv", "w+")
        f.write("epoch;nb_episodes;r_mean;r_var;r_std\n")
        f.close()

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        print(self._periodicity)

        mod = self._epoch_count % self._periodicity
        self._epoch_count += 1
        if mod == 0:
            agent.startMode(self._id, self._epoch_length)
            agent.setControllersActive(self._to_disable, False)
        elif mod == 1:
            self._summary_counter += 1
            n_tests = self._number_tests
            
            scores = []
            nb_episodes = 0
            while n_tests > 0 : 
                if self._show_score:
                    score,nbr_episodes=agent.totalRewardOverLastTest()
                    scores.append(score)
                    nb_episodes += nbr_episodes
                    print("Testing score per episode (id: {}) is {} (average over {} episode(s))".format(self._id, score, nbr_episodes))
                if self._summary_periodicity > 0 and self._summary_counter % self._summary_periodicity == 0:
                    try:
                        os.makedirs(self._path_files + "/" + "epoch_" + str(self._epoch_count))
                    except:
                        pass

                    agent.summarizeTestPerformance(path_dump=self._path_files + "/" + "epoch_" + str(self._epoch_count),prefix_file=self._prefix_file)
                agent._training_loss_averages = []
                agent.startMode(self._id, self._epoch_length)
                if agent._mode != -1:
                    agent._totalModeNbrEpisode=0
                    while agent._mode_epochs_length > 0:
                        agent._totalModeNbrEpisode += 1
                        agent._mode_epochs_length = agent._runEpisode(agent._mode_epochs_length)
                n_tests -= 1 
            f = open(self._path_files + "/" + self._prefix_file + "_summary.csv", "a+")
            f.write(str(self._epoch_count) + ";" + str(nb_episodes) + ";" + str(np.mean(np.asarray(scores)))+";"+str(np.var(np.asarray(scores)))+";"+str(np.std(np.asarray(scores)))+"\n")
            f.close()
            agent.resumeTrainingMode()
            agent.setControllersActive(self._to_disable, True)

            


