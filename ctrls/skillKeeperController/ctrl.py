import numpy as np
import joblib
import os
from ctrls.controller import Controller

class SkillKeeperController(Controller):
    """[Experimental] A controller that tries to avoid unstabilities by changing loss function on the fly. Whenever a "best" candidate model is spot, the loss function
       is changed to a weighted one to take into account mse + kl divergence of the current learning model to the candidate model. 
       It has to be called *before* actual training.
    
    Parameters
    ----------
    w_mse : float 
        Initial weight for mse
    w_kld : float 
        Initial weight of kld
    w_mse_decay : float 
        Weight decay for w_mse
    w_kld_decay : float 
        Weight decay for w_kld
    """

    def __init__(self, evaluate_on="action",w_mse = 0.1, w_kld = 0.9, w_decays = 10000):
        super(self.__class__, self).__init__()
        self._w_mse = w_mse
        self._w_kld = w_kld
        self._w_decay_mse = 0 if w_decays == 0 else (1 - w_mse) / w_decays
        self._w_decay_kld = 0 if w_decays == 0 else (w_kld) / w_decays
        self._ws = [w_mse]
        self._on_action = "action" == evaluate_on
        self._on_episode = "episode" == evaluate_on
        self._on_epoch = "epoch" == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

    def _perform(self,agent):
        networks = agent.getNetworks()
        
        if len(networks) == 0:
            return
        diff = len(networks) - len(self._ws)
        for i in range(1,len(self._ws)):
                #Decrease w_kld
                self._ws[i] = max(0,self._ws[i] - self._w_mse_decay)
        for i in range(diff):
                self._ws.append(self._w_kld)

        #Get data batch and store it in agent
        states,_,_,_,_,_ = agent.generateAndStoreBatch()
        q_targs = []
        w_klds = []
        for i in range(len(networks)):
                network = networks[i]
                w_kld = self._ws[i]
                if w_kld > 0:
                    q_targs.append(network.batchPredict(states))
                w_klds.append(w_kld)
        agent._network._compile("skillkeeper_loss",w_klds,q_targs)


    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return
        
        if self._on_episode:
            self._perform(agent)


    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._on_epoch:
            self._perform(agent)

    def onActionTaken(self, agent):
        if (self._active == False):
            return

        if self._on_action:
            self._perform(agent)
