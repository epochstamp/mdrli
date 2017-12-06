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

    def __init__(self, validationID=0, testID=None, unique_fname="nnet"):
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

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        mode = agent.mode()
        if mode == self._validationID:
            score, _ = agent.totalRewardOverLastTest()
            self._validationScores.append(score)
            self._epochNumbers.append(self._trainingEpochCount)
            if score > self._bestValidationScoreSoFar:
                self._bestValidationScoreSoFar = score
                agent.dumpNetwork(self._filename, self._trainingEpochCount)
        elif mode == self._testID:
            score, _ = agent.totalRewardOverLastTest()
            self._testScores.append(score)
        else:
            self._trainingEpochCount += 1
        
    def onEnd(self, agent):
        if (self._active == False):
            return

        bestIndex = np.argmax(self._validationScores)
        print("Best neural net obtained after {} epochs, with validation score {}".format(bestIndex+1, self._validationScores[bestIndex]))
        if self._testID != None:
            print("Test score of this neural net: {}".format(self._testScores[bestIndex]))
                
        try:
            os.mkdir("scores")
        except Exception:
            pass
        basename = "scores/" + self._filename
        joblib.dump({"vs": self._validationScores, "ts": self._testScores}, basename + "_scores.jldump")
