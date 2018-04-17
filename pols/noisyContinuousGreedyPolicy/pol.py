from pols.modelpolicy import ModelPolicy
from joblib import load
import numpy as np

def rescale(x,minx,maxx):
    return ((maxx - minx)/(2)) * (x-1) + maxx

def noise_function(name,rng,n_actions):
       
    def zero(u,noise_scale):
        return u

    def linear(u,noise_scale):
        return u + rng.randn(n_actions) / (noise_scale)

    def exp(u,noise_scale):
        return u + rng.randn(n_actions) * 10 ** (-noise_scale)

    def fixed(u,noise_scale):
        return u + rng.randn(n_actions) * noise_scale

    def covariance(u,noise_scale,px):
        if px is None : return 0
        if n_actions == 1:
            std = np.minimum(noise_scale / px[0], 1)[0]
            action = rng.normal(u,std,size=(1,))
        else:
            cov = np.minimum(np.linalg.inv(px[0]) * noise_scale, 1)
            action = rng.multivariate_normal(u, cov)
        return action

    try:
        return eval(name)
    except Exception as e:
        print("Warning : The name does not refer to a valid mapping between name and noise function. Output of the exception : ")
        print(e)        

class NoisyContinuousGreedyPolicy(ModelPolicy):

    def __init__(self, n_actions,random_state, noise_func='zero', noise_coeff=0):
        ModelPolicy.__init__(self,n_actions,random_state)  
        self.noise_name = noise_func   
        self.noise_func = noise_function(noise_func,random_state,len(n_actions))
        self.noise_coeff = noise_coeff
        self.P = None




    def action(self, state):
        if not hasattr(self,"_model") or self._model is None:
            raise AttributeError("Model has not been set in this policy")
        try:
            action, _ = self._model.chooseBestAction(state)
        except Exception as e:
            raise AttributeError("Model does not meet required specifications or is corrupted. Here is the error message : " + str(e))

        if self.P is None and self.noise_name == "covariance":
            self.P = self._model.P 
        if self.noise_name == "covariance":
            new_action = self.noise_func(action,self.noise_coeff, self.P([np.expand_dims(s,axis=0) for s in state]))
        else:
            new_action = self.noise_func(action,self.noise_coeff)
        V = self._model.evalAction(state,new_action)
        new_action = np.clip(new_action,self.n_actions[0][0],self.n_actions[0][1])
        #new_action = list(map(lambda i : rescale(new_action[i],self.n_actions[i][0],self.n_actions[i][1]), range(len(self.n_actions))))
        #print (new_action)
        if self.noise_name == "zero" :
            print (state) 
            print (new_action)
        return new_action, V


    def setAttribute(self,attr,value):
        ModelPolicy.setAttribute(self,attr,value)
        if attr=="noise_coeff":
            self.noise_coeff = value
        if attr=="noise_func":
            self.noise_func = value

    def getAttribute(self,attr):
        return ModelPolicy.setAttribute(self,attr,value)
        if attr=="noise_coeff":
            return self.noise_coeff
        if attr=="noise_func":
            return self.noise_func
