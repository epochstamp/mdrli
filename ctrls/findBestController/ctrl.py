import numpy as np
import joblib
import os
from ctrls.controller import Controller

class FindBestController(Controller):
    """A controller that finds the neural net performing at best in validation mode (i.e. for mode = [validationID]) 
    and computes the associated generalization score in test mode (i.e. for mode = [testID], and this only if [testID] 
    is different from None). This controller should never be disabled by InterleavedTestControllers as it is meant to 
    work in conjunction with them.
    
    At each epoch end where this controller is active, it will look at the current mode the agent is in. 
    
    If the mode matches [validationID], it will take the total reward of the agent on this epoch and compare it to its 
    current best score. If it is better, it will ask the agent to dump its current nnet on disk and update its current 
    best score. In all cases, it saves the validation score obtained in a vector.

    If the mode matches [testID], it saves the test (= generalization) score in another vector. Note that if [testID] 
    is None, no test mode score are ever recorded.

    At the end of the experiment (onEnd), if active, this controller will print information about the epoch at which 
    the best neural net was found together with its generalization score, this last information shown only if [testID] 
    is different from None. Finally it will dump a dictionnary containing the data of the plots ({n: number of 
    epochs elapsed, ts: test scores, vs: validation scores}). Note that if [testID] is None, the value dumped for the
    'ts' key is [].
    
    Parameters
    ----------
    validationID : int 
        See synopsis
    testID : int 
        See synopsis
    unique_fname : str
        A unique filename (basename for score and network dumps).
    """

    def __init__(self, validationID=0, testID=None, path_dump=".",unique_fname="nnet"):
        super(self.__class__, self).__init__()
        validationID = int(validationID)
        testID = None if testID is None else int(testID)
        self._validationScores = []
        self._testScores = []
        self._epochNumbers = []
        self._trainingEpochCount = 0
        self._testID = testID
        self._validationID = validationID
        self._filename = unique_fname
        self._bestValidationScoreSoFar = -9999999	
        self._path_dump = path_dump
        try:
            os.makedirs(self._path_dump + "/scores/")
        except Exception:
            pass

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        mode = agent.mode()
        if mode == self._validationID:
            mean_score,_,std_score,_ = agent.statRewardsOverLastTests()
            self._validationScores.append((mean_score,std_score))
            self._epochNumbers.append(self._trainingEpochCount)
            if mean_score - std_score > self._bestValidationScoreSoFar:
                self._bestValidationScoreSoFar = mean_score - std_score
                agent.dumpNetwork(self._filename, self._trainingEpochCount,self._path_dump)
                agent.storeNetwork()
        elif mode == self._testID:
            mean_score,_,std_score,_ = agent.statRewardsOverLastTests()
            self._testScores.append((mean_score,std_score))
        else:
            self._trainingEpochCount += 1
        
    def onEnd(self, agent):
        if (self._active == False):
            return

        bestIndex = np.argmax(list(map(lambda x : x[0] - x[1],self._validationScores)))
        print("Best neural net obtained after {} epochs, with validation score {} (std : {})".format(bestIndex+1, self._validationScores[bestIndex][0],self._validationScores[bestIndex][1]))
        if self._testID != None:
            print("Test score of this neural net: {} (std : {})".format(self._testScores[bestIndex][0],self._testScores[bestIndex][1]))
                

        basename = self._path_dump + "scores/" + self._filename
        joblib.dump({"vs": self._validationScores, "ts": self._testScores}, basename + "_scores.jldump")
