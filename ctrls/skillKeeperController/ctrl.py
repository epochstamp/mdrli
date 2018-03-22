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

    def __init__(self, evaluate_on="action", t_kld = 100., t_decays = 2., periodicity = 2, skillkeeper_mode=0):
        super(self.__class__, self).__init__()
        self._t_kld = float(t_kld)
        self._t_decay = t_decays
        self._ts = []
        self._on_action = "action" == evaluate_on
        self._on_episode = "episode" == evaluate_on
        self._on_epoch = "epoch" == evaluate_on
        self._networks = []
        self._previous_size = 0
        self._periodicity = periodicity

        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

    def _perform(self,agent):
        if agent.mode() != -1:
            return

        

        self._count += 1
        if self._count % self._periodicity == 0:
            networks = agent.getNetworks()
            offset = len(networks) - self._previous_size
            self._previous_size = len(networks) - offset
            if len(networks) == 0:
                return
            #Select the best network
            network, score = networks[offset-1]
            bestScore = score["score"]
            for i in range(offset-1,len(networks)):
                   n,s = networks[i]
                   score = s["score"]
                   if score > bestScore:
                        bestScore = score
                        network = n     
            
            self._networks.append(network)
            
            print("New neural network found")
            for i in range(len(self._ts)):
                    self._ts[i] /= float(self._t_decay)
            self._ts.append(float(self._t_kld))
            #Get data batch and store it in agent
            states,_,rewards,next_states,terminals,_ = agent.generateAndStoreBatch()
            q_targs = []
            t_klds = []
            q_next_targs = []
            for i in range(len(self._networks)):
                    network = self._networks[i]
                    t_kld = float(self._ts[i])
                    if t_kld > 0.00001:
                        q_targs.append(network.batchPredict(states))
                        q_next_targs.append(network.batchPredict(next_states))
                        t_klds.append(t_kld)
                    else:
                        self._ts.remove(i)
                        
            agent._network._compile("skillkeeper_loss",skillkeeper_mode, rewards,agent.discountFactor(),t_klds,q_targs,q_next_targs)


    def onStart(self,agent):
        if (self._active == False):
            return
        self._count = 0

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
