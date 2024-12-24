import numpy as np

import os
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union

from model.surrogate_model import SurrogateModel

import matplotlib.pyplot as plt
from scipy.stats import qmc

from sklearn.metrics import mean_squared_error

type Vector = List[float]

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
        model: Literal["NN", "smth", "smth"] = "NN",
        model_param: dict = None,
        rnd_state: int = 32,
        verbose: bool = False
    ):
        
        # init model
        self.model: SurrogateModel
        self.surrogate_name: str = model

        model_param: dict = {} if model_param is None else model_param

        self.model1 = self.init_model(
            self.surrogate_name,
            **model_param
        )
        self.model2 = self.init_model(
            self.surrogate_name,
            **model_param
        )
        self.model3 = self.init_model(
            self.surrogate_name,
            **model_param
        )
        self.model4 = self.init_model(
            self.surrogate_name,
            **model_param
        )
        self.model5 = self.init_model(
            self.surrogate_name,
            **model_param
        )
        
        assert self.model1 and self.model2 and self.model3 and self.model4 and self.model5 is not None

        self.func = func

        self.prev_loss = 0.

        self.max_iter = 150
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

        self.pred1 = []
        for m in [self.model1,self.model2,self.model3,self.model4,self.model5]:
            m.fit(self.X_init,self.y_init,first = True)
            self.pred1.append(m.predict(self.test_set).flatten())
        self.pred1 = np.sum(self.pred1,axis = 0)

    def init_model(self,
                   model_name: Literal["NN", "smth"],
                   **model_args,):
        """
        Initialize model

        """
        if model_name == "NN":
            from model.NN import NN_model

            return NN_model(**model_args)
       
    def Step(self,
             a: np.ndarray):

        self.cur_iter += 1
        truncated = False
        terminated = False

        self.X = np.concatenate((self.X,[a]),axis=0)
        self.y = np.concatenate((self.y,[self.func(a)]),axis=0)

        for m in [self.model1,self.model2,self.model3,self.model4,self.model5]:
            m.fit(self.X,self.y)

        #_,model_state = self.model.fit(self.X,self.y)

        #self.prev_loss = self.model.loss

        #state = np.concatenate((model_state, self.model.predict(self.test_set).flatten(), self.X[-10:].flatten()))

        pred2 = []
        for m in [self.model1,self.model2,self.model3,self.model4,self.model5]:
            pred2.append(m.predict(self.test_set).flatten())

        state = np.concatenate((np.var(pred2,axis = 0), self.X[-20:].flatten()))
        
            
        reward = mean_squared_error(self.test_set_y,np.mean(pred2,axis = 0))

        if self.cur_iter == self.max_iter:
            self.cur_iter = 0
            truncated = True
            
        if reward < 0.001:
            self.cur_iter = 0
            terminated = True
            reward = -1

        #print(state,reward,terminated,truncated)
        
        #return state,reward,terminated,truncated
        return state,-reward,terminated,truncated

    def Reset(self):

        #model_state = self.model1.Reset()
        for m in [self.model1,self.model2,self.model3,self.model4,self.model5]:
            m.Reset()
       
        self.X = self.X_init
        self.y = self.y_init
        self.cur_iter = 0

        #state = np.concatenate((model_state,self.model.predict(self.test_set).flatten(), self.X[-10:].flatten()))
        #state = np.concatenate((self.model.predict(self.test_set).flatten(), self.X[-10:].flatten()))
        preds = np.var([self.model1.predict(self.test_set).flatten(),self.model2.predict(self.test_set).flatten(),\
                                self.model3.predict(self.test_set).flatten(),self.model4.predict(self.test_set).flatten(),\
                                self.model5.predict(self.test_set).flatten()], axis=0)

        state = np.concatenate((preds,self.X[-20:].flatten()))
        
        return state

    def toy_Plot(self,x):

        fig, ax = plt.subplots(figsize=(15, 3))

        #y_pred = self.model.predict(x.reshape(-1,1)).flatten()
        
        y_pred = []
        for m in [self.model1,self.model2,self.model3,self.model4,self.model5]:
            y_pred.append(m.predict(x.reshape(-1,1)))
        y_pred = np.mean(y_pred,axis = 0).flatten()


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