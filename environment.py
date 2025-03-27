import numpy as np

import os
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union

from model.surrogate_model import SurrogateModel

import matplotlib.pyplot as plt
from scipy.stats import qmc

from sklearn.metrics import mean_squared_error
from copy import deepcopy

class Environment:
    """
    Class for environment

    Parameters
    ----------
    model: str, optional(default="NN") 
    model_param: dict, optional(default=None) additional arguments used to creating the surrogate model
    
    verbose: bool, optional (default=False) plot information during iterations
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l_bounds: np.ndarray,
        u_bounds: np.ndarray,
        func: Callable[[np.ndarray, ...], np.ndarray],
        model: Literal["NN", "NN_tf", "GP"] = "NN",
        model_param: dict = None,
        state_mode: str = 'last_points',
        N: int = 50,
        rnd_state: int = 32,
        verbose: bool = False
    ):
        
        # init model
        self.state_mode = state_mode
        self.seed = 32
        self.model: SurrogateModel
        self.surrogate_name: str = model

        model_param: dict = {} if model_param is None else model_param

        self.m = self.init_model(
            self.surrogate_name,
            **model_param
        )

        self.l_bounds=l_bounds
        self.u_bounds=u_bounds

        self.func = func

        self.prev_loss = 0.

        self.max_iter = N
        self.cur_iter = 0

        #initial training set
        self.y_init = y.reshape(-1,1)
        self.X_init = X.reshape(len(self.y_init),-1)

        #extended training set
        self.X = self.X_init
        self.y = self.y_init

        sampler = qmc.LatinHypercube(d=self.X.shape[1])
        sample = sampler.random(n=100)
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        self.test_set = np.array(sample_scaled)
        self.test_set_y = func(self.test_set)

        sample = sampler.random(n=20)
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        self.state_set = np.sort(np.array(sample_scaled),axis=0)
    

        self.pred1 = []
       
        self.m.fit(self.X_init,self.y_init,first = True)
        print(self.m.params())
        
        pred = self.m.predict(self.test_set)
        self.mse_loss = mean_squared_error(self.test_set_y,pred)
            
        self.pred = self.m.predict(self.test_set)
        self.N = N

    def init_model(self,
                   model_name: Literal["NN", "NN_tf","GP"],
                   **model_args,):

        if model_name == "NN":
            from models.NN import NN_model

            return NN_model(**model_args)
            
        if model_name == "NN_tf":
            from models.NN_tf import NN_tf_model

            return NN_tf_model(**model_args)

        if model_name == "GP":
            from models.GP import GP_model

            return GP_model(**model_args)
       
    def step(self,
             a: np.ndarray):

        self.cur_iter += 1
        truncated = False
        terminated = False

        self.pred_X = deepcopy(self.X)
        self.pred_y = deepcopy(self.y)

        self.X = np.concatenate(([a],self.X),axis=0)
        self.y = np.concatenate(([[self.func(a)]],self.y),axis=0)
        
        self.pred_m = deepcopy(self.m)

        self.m.fit(self.X[:-self.N+1],self.y[:-self.N+1])

        pred = self.m.predict(self.test_set)

        if self.state_mode == 'last_points':
            if self.cur_iter<self.N:
                state = self.X[:self.N].flatten()
            else:
                state = self.X[:self.N].flatten()
                
        elif self.state_mode == 'predictions+last_points':
            if self.cur_iter<self.N:
                state = np.concatenate((self.m.predict(self.state_set), self.X[-self.cur_iter:].flatten(),self.X[:self.N-self.cur_iter].flatten()))
            else:
                state = np.concatenate((self.m.predict(self.state_set), self.X[-self.N:].flatten()))
                
        elif self.state_mode == 'model_parameters':
            pass
                    
        self.mse_loss = mean_squared_error(self.test_set_y,pred)
        
        L2 = mean_squared_error(pred,self.pred)  
        self.pred = pred

        reward = L2

        if self.cur_iter == self.max_iter:
                self.cur_iter = 0
                truncated = True

        if not truncated and self.cur_iter>1:
            
            if np.abs(L2-self.prev_reward) < 0.01:
                
                flag = 0
                sample_scaled = np.linspace(-1,1,10).reshape(-1,1)
                
                for a in sample_scaled:
                    if np.sqrt(mean_squared_error(self.m.predict([a]),[self.func(a)]))>0.08:
                        flag = 1
                        break
               
                if flag == 0:
                    self.cur_iter = 0
                    terminated = True
                    reward = 1
                
        self.prev_reward = L2
        self.observation_space = np.array(state)
        self.action_space = np.array([a])

     
        return state,reward,terminated,truncated

    def reset(self):

        self.m.Reset()
       
        self.X = self.X_init
        self.y = self.y_init
        self.cur_iter = 0

        preds = self.m.predict(self.state_set)
        
        if self.state_mode == 'last_points':
            state = self.X[:self.N].flatten()
                
        elif self.state_mode == 'predictions+last_points':
            state = np.concatenate((preds,self.X[-self.N:].flatten()))
                
        elif self.state_mode == 'model_parameters':
            pass

        self.observation_space = np.array(state)
        self.action_space = np.array([0])

        self.pred = self.m.predict(self.test_set)
        
        return state
  

    def toy_Plot(self,x):

        fig, ax = plt.subplots(figsize=(15, 3))
        
        y_pred = []
        for m in [self.m]:
            y_pred.append(m.predict(x.reshape(-1,1)))
        y_pred = m.predict(x.reshape(-1,1))


        ax.plot()
        y_test = self.func(x).flatten()

        ax.plot(x, y_test, label=r"Target $f$", ls="--", color="k", lw=2)
        ax.plot(x, y_pred, label=r"Model $\hat{f}$", lw=2)


        ax.plot(
          self.X,
          self.y,
          "o",
          color="g",
          markersize=8,
          label="New Sample",
        )

        ax.plot(
          self.X_init,
          self.y_init,
          "o",
          color="black",
          markersize=8,
          label="Initial Sample",
        )


        ax.set_ylabel(r"$\mathrm{Y}$")
        plt.legend()
        plt.tight_layout()
        plt.show()
