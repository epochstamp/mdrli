"""This module contains classes used to define the standard behavior of the agent.
It relies on the controllers, the chosen training/test policy and the learning algorithm
to specify its behavior in the environment.

.. Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
import os
import numpy as np
import copy
import sys
import joblib
from warnings import warn
from data.dataset import DataSet,SliceError
from ctrls.controller import Controller
from deer.helper import tree 
from pols.epsilonGreedyPolicy.pol import EpsilonGreedyPolicy

class NeuralAgent(object):
    """The NeuralAgent class wraps a deep Q-network for training and testing in a given environment.
    
    Attach controllers to it in order to conduct an experiment (when to train the agent, when to test,...).
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent interacts
    q_network : object from class QNetwork
        The q_network associated to the agent
    replay_memory_size : int
        Size of the replay memory. Default : 1000000
    replay_start_size : int
        Number of observations (=number of time steps taken) in the replay memory before starting learning. 
        Default: minimum possible according to environment.inputDimensions().
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    random_state : numpy random number generator
        Default : random seed.
    exp_priority : float
        The exponent that determines how much prioritization is used, default is 0 (uniform priority).
        One may check out Schaul et al. (2016) - Prioritized Experience Replay.
    train_policy : object from class Policy
        Policy followed when in training mode (mode -1)
    test_policy : object from class Policy
        Policy followed when in other modes than training (validation and test modes)
    only_full_history : boolean
        Whether we wish to train the neural network only on full histories or we wish to fill with zeroes the 
        observations before the beginning of the episode
    """

    def __init__(self, environments, q_networks, replay_memory_size=1000000, replay_start_size=None, batch_size=32, random_state=np.random.RandomState(), exp_priority=0, train_policy=None, test_policy=None, only_full_history=True,init_env=0):
        
        
        self._controllers = []
        self._environments = environments
        self._networks = q_networks
        self._e = init_env
        self._environment = environments[self._e]
        self._network = self._networks[self._e]
        self._replay_memory_size = replay_memory_size
        self._replay_start_size = replay_start_size
        self._batch_size = batch_size
        self._random_state = random_state
        self._exp_priority = exp_priority
        self._only_full_history = only_full_history
        self._datasets = list()
        for i in range(len(self._environments)):
            self._datasets.append(DataSet(self._environments[i], max_size=replay_memory_size, random_state=random_state, use_priority=self._exp_priority, only_full_history=self._only_full_history))
        self._dataset = self._datasets[self._e]
        self._tmp_dataset = None # Will be created by startTesting() when necessary
        self._mode = -1
        self._mode_epochs_length = 0
        self._total_mode_reward = 0
        self._training_loss_averages = []
        self._Vs_on_last_episode = []
        self._in_episode = False
        self._selected_action = -1

        self._states = [None] * len(self._environments)
        for i in range(len(self._environments)):
            self._states[i] = []
            inputDims = self._environments[i].inputDimensions()
        
            if replay_start_size is None:
                replay_start_size = max(inputDims[i][0] for i in range(len(inputDims)))
            elif replay_start_size < max(inputDims[i][0] for i in range(len(inputDims))) :
                raise AgentError("Replay_start_size should be greater than the biggest history of a state.")
            for j in range(len(inputDims)):
                self._states[i].append(np.zeros(inputDims[j], dtype=config.floatX))
            if (train_policy==None):
                self._train_policy = EpsilonGreedyPolicy(self._environments[i].nActions(), random_state, 0.1)
                self._train_policy.setAttribute("model",q_networks[i])
            else:
                #Todo : change the number of actions. Listify the policies
                self._train_policy = train_policy
            if (test_policy==None):
                self._test_policy = EpsilonGreedyPolicy(self._environments[i].nActions(), random_state, 0.)
                self._test_policy.setAttribute("model",q_networks[i])
            else:
                #Todo : change the number of actions
                self._test_policy = test_policy
        self._state = self._states[self._e]

    def setEnvironment(self,e,reset=False):
        """ Change the environment and the related dataset
        """
        self._e = e
        self._dataset = self._datasets[self._e]
        self._state = self._states[e]
        if reset:
            self._state[...] = 0
            self._dataset.flush()
            if self._tmp_dataset is not None:
                self._tmp_dataset.flush()
        self._environment = self._environments[self._e]

    def setControllersActive(self, toDisable, active):
        """ Activate controller
        """
        for i in toDisable:
            self._controllers[i].setActive(active)

    def setLearningRate(self, lr):
        """ Set the learning rate for the gradient descent
        """
        self._network.setLearningRate(lr)

    def learningRate(self):
        """ Get the learning rate
        """
        return self._network.learningRate()

    def setDiscountFactor(self, df):
        """ Set the discount factor
        """
        self._network.setDiscountFactor(df)

    def discountFactor(self):
        """ Get the discount factor
        """
        return self._network.discountFactor()

    def overrideNextAction(self, action):
        """ Possibility to override the chosen action. This possibility should be used on the signal OnActionChosen.
        """
        self._selected_action = action

    def avgBellmanResidual(self):
        """ Returns the average training loss on the epoch
        """
        if (len(self._training_loss_averages) == 0):
            return -1
        return np.average(self._training_loss_averages)

    def avgEpisodeVValue(self):
        """ Returns the average V value on the episode (on time steps where a non-random action has been taken)
        """
        if (len(self._Vs_on_last_episode) == 0):
            return -1
        if(np.trim_zeros(self._Vs_on_last_episode)!=[]):
            return np.average(np.trim_zeros(self._Vs_on_last_episode))
        else:
            return 0

    def totalRewardOverLastTest(self):
        """ Returns the average sum of rewards per episode and the number of episode
        """
             
        return self._total_mode_reward/self._totalModeNbrEpisode, self._totalModeNbrEpisode

    def statRewardsOverLastTests(self):
        """ Returns the average sum of rewards per episode and the number of episode
        """
           
        return np.mean(self._mode_rewards),np.var(self._mode_rewards),np.std(self._mode_rewards), self._totalModeNbrEpisode

    def bestAction(self):
        """ Returns the best Action
        """
        action = self._network.chooseBestAction(self._state)
        V = max(self._network.qValues(self._state))
        return action, V
     
    def attach(self, controller):
        if (isinstance(controller, Controller)):
            self._controllers.append(controller)
        else:
            raise TypeError("The object you try to attach is not a Controller.")

    def detach(self, controllerIdx):
        return self._controllers.pop(controllerIdx)

    def mode(self):
        return self._mode

    def startMode(self, mode, epochLength,n_episodes=None):
        if self._in_episode:
            raise AgentError("Trying to start mode while current episode is not yet finished. This method can be "
                             "called only *between* episodes for testing and validation.")
        elif mode == -1:
            raise AgentError("Mode -1 is reserved and means 'training mode'; use resumeTrainingMode() instead.")
        else:
            self._n_episodes = self._n_episodes_init if n_episodes is None else n_episodes 
            self._mode = mode
            self._mode_epochs_length = epochLength
            self._total_mode_reward = 0.
            del self._tmp_dataset
            self._tmp_dataset = DataSet(self._environment, self._random_state, max_size=self._replay_memory_size, only_full_history=self._only_full_history)

    def resumeTrainingMode(self,n_episodes = None):
        self._n_episodes = self._n_episodes_init if n_episodes is None else self._n_episodes
        self._mode = -1

    def summarizeTestPerformance(self, **kwargs):
        if self._mode == -1:
            raise AgentError("Cannot summarize test performance outside test environment.")

        self._environment.summarizePerformance(self._tmp_dataset,**kwargs)

    def train(self):
        """
        This function selects a random batch of data (with self._dataset.randomBatch) and performs a 
        Q-learning iteration (with self._network.train).        
        """
        # We make sure that the number of elements in the replay memory
        # is strictly superior to self._replay_start_size before taking 
        # a random batch and perform training
        if self._dataset.n_elems <= self._replay_start_size:
            return

        try:
            states, actions, rewards, next_states, terminals, rndValidIndices = self._dataset.randomBatch(self._batch_size, self._exp_priority)
            loss, loss_ind = self._network.train(states, actions, rewards, next_states, terminals)
            self._training_loss_averages.append(loss)
            if (self._exp_priority):
                self._dataset.updatePriorities(pow(loss_ind,self._exp_priority)+0.0001, rndValidIndices[1])

        except SliceError as e:
            warn("Training not done - " + str(e), AgentWarning)

    def dumpNetwork(self,fname, nEpoch=-1, path="."):
        """ Dump the network
        
        Parameters
        -----------
        fname : string
            Name of the file where the network will be dumped
        nEpoch : int
            Epoch number (Optional)
        """
        try:
            os.makedirs(path + "/nnets")
        except Exception:
            pass
        basename = path + "/nnets/" + fname

        for f in os.listdir(path + "/nnets/"):
            if fname in f:
                os.remove(path +  "/nnets/" + f)

        all_params = self._network.getAllParams()

        if (nEpoch>=0):
            joblib.dump(all_params, basename + ".epoch={}".format(nEpoch))
        else:
            joblib.dump(all_params, basename, compress=True)

    def setNetwork(self, fname, nEpoch=-1):
        """ Set values into the network
        
        Parameters
        -----------
        fname : string
            Name of the file where the values are
        nEpoch : int
            Epoch number (Optional)
        """

        basename = "nnets/" + fname

        if (nEpoch>=0):
            all_params = joblib.load(basename + ".epoch={}".format(nEpoch))
        else:
            all_params = joblib.load(basename)

        self._network.setAllParams(all_params)

    def run(self, n_epochs, epoch_length, n_episodes = 1):
        """
        This function encapsulates the whole process of the learning.
        It starts by calling the controllers method "onStart", 
        Then it runs a given number of epochs where an epoch is made up of one or many episodes (called with 
        agent._runEpisode) and where an epoch ends up after the number of steps reaches the argument "epoch_length".
        It ends up by calling the controllers method "end".

        Parameters
        -----------
        n_epochs : number of epochs 
            int
        epoch_length : maximum number of steps for a given epoch
            int
        """
        self._n_episodes = n_episodes
        self._n_episodes_init = n_episodes
        for c in self._controllers: c.onStart(self)
        i = 0
        while i < n_epochs or self._mode_epochs_length > 0:
            self._training_loss_averages = []
            
            if self._mode != -1:
                self._totalModeNbrEpisode=0
                self._mode_rewards = []
                
                while self._totalModeNbrEpisode < self._n_episodes:
                    mode_epoch_length = self._mode_epochs_length 
                    while mode_epoch_length > 0:
                        self._totalModeNbrEpisode += 1
                        mode_epoch_length = self._runEpisode(mode_epoch_length)
                    
                    self._mode_rewards.append(self._total_mode_reward)
                    self._total_mode_reward = 0
                self._mode_epochs_length = 0 
            else:
                length = epoch_length
                n_episodes = self._n_episodes
                while n_episodes > 0:               
                    while length > 0:
                        length = self._runEpisode(length)
                    n_episodes -= 1
                i += 1
            for c in self._controllers: c.onEpochEnd(self)
            
        self._environment.end()
        for c in self._controllers: c.onEnd(self)

    def _runEpisode(self, maxSteps):
        """
        This function runs an episode of learning. An episode ends up when the environment method "inTerminalState" 
        returns True (or when the number of steps reaches the argument "maxSteps")
        
        Parameters
        -----------
        maxSteps : maximum number of steps before automatically ending the episode
            int
        """
        self._in_episode = True
        initState = self._environment.reset(self._mode)
        inputDims = self._environment.inputDimensions()
        for i in range(len(inputDims)):
            if inputDims[i][0] > 1:
                self._state[i][1:] = initState[i][1:]
        
        self._Vs_on_last_episode = []
        while maxSteps > 0:
            maxSteps -= 1

            obs = self._environment.observe()

            for i in range(len(obs)):
                self._state[i][0:-1] = self._state[i][1:]
                self._state[i][-1] = obs[i]

            V, action, reward = self._step()
            
            self._Vs_on_last_episode.append(V)
            if self._mode != -1:
                self._total_mode_reward += reward

            is_terminal = self._environment.inTerminalState() or maxSteps == 0
                
            self._addSample(obs, action, reward, is_terminal)
            for c in self._controllers: c.onActionTaken(self)
            
            if is_terminal:
                break
            
        self._in_episode = False
        for c in self._controllers: c.onEpisodeEnd(self, is_terminal, reward)
        return maxSteps

        
    def _step(self):
        """
        This method is called at each time step. If the agent is currently in testing mode, and if its *test* replay 
        memory has enough samples, it will select the best action it can. If there are not enough samples, FIXME.
        In the case the agent is not in testing mode, if its replay memory has enough samples, it will select the best 
        action it can with probability 1-CurrentEpsilon and a random action otherwise. If there are not enough samples, 
        it will always select a random action.
        Parameters
        -----------
        state : ndarray
            An ndarray(size=number_of_inputs, dtype='object), where states[input] is a 1+D matrix of dimensions
               input.historySize x "shape of a given ponctual observation for this input".
        Returns
        -------
        action : int
            The id of the action selected by the agent.
        V : float
            Estimated value function of current state.
        """

        action, V = self._chooseAction()        
        reward = self._environment.act(action)

        return V, action, reward

    def _addSample(self, ponctualObs, action, reward, is_terminal):
        if self._mode != -1:
            self._tmp_dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1)
        else:
            self._dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1)


    def _chooseAction(self):
        
        if self._mode != -1:
            # Act according to the test policy if not in training mode
            action, V = self._test_policy.action(self._state)
        else:
            if self._dataset.n_elems > self._replay_start_size:
                # follow the train policy
                action, V = self._train_policy.action(self._state)     #is self._state the only way to store/pass the state?
            else:
                # Still gathering initial data: choose dummy action
                action, V = self._train_policy.randomAction()
                
        for c in self._controllers: c.onActionChosen(self, action)
        return action, V

class AgentError(RuntimeError):
    """Exception raised for errors when calling the various Agent methods at wrong times.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class AgentWarning(RuntimeWarning):
    """Warning issued of the various Agent methods.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """


