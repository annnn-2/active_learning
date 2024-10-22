import numpy as np

import os
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union

from model.surrogate_model import SurrogateModel

import matplotlib.pyplot as plt
from scipy.stats import qmc

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

        self.model = self.init_model(
            self.surrogate_name,
            **model_param
        )
        print(self.model)
        assert self.model is not None

        self.func = func

        self.prev_loss = 0.

        self.max_iter = 20
        self.cur_iter = 0

        #initial training set
        self.y_init = y.reshape(-1,1)
        self.X_init = X.reshape(len(self.y_init),-1)

        #extended training set
        self.X = self.X_init
        self.y = self.y_init

        sampler = qmc.LatinHypercube(d=self.X.shape[1])
        sample = sampler.random(n=1000)
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        self.test_set = np.array(sample_scaled)
        
        self.model.fit(self.X_init,self.y_init,first = True)

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
        truncated = True
        terminated = True

        self.X = np.concatenate((self.X,[a]),axis=0)
        self.y = np.concatenate((self.y,[self.func(a)]),axis=0)

        _,model_state = self.model.fit(self.X,self.y)
        #self.model.fit(self.X,self.y)

        if self.cur_iter == self.max_iter:
            self.cur_iter = 0
            truncated = False

        if (self.prev_loss - self.model.loss) < 0.000001:
            terminated = False

        self.prev_loss = self.model.loss


        state = np.concatenate((model_state, self.model.predict(self.test_set).flatten(), self.X[-10:].flatten()))
        #print(state)
        
        return state,-self.model.loss,terminated,truncated

    def Reset(self):

        model_state = self.model.Reset()
        self.model.Reset()
        self.X = self.X_init
        self.y = self.y_init

        state = np.concatenate((model_state,self.model.predict(self.test_set).flatten(), self.X[-10:].flatten()))
        
        return state

    def toy_Plot(self,x):

        fig, ax = plt.subplots(figsize=(15, 3))

        y_pred = self.model.predict(x.reshape(-1,1)).flatten()


        ax.plot()
        y_test = self.func(x).flatten()

        ax.plot(x, y_test, label=r"Target $f$", ls="--", color="k", lw=2)
        ax.plot(x, y_pred, label=r"Model $\hat{f}$", lw=2)

        ax.plot(
          self.X_init,
          self.y_init,
          "o",
          color="k",
          markersize=8,
          label="Initial Sample",
        )

        ax.plot(
          self.X,
          self.y,
          "o",
          color="g",
          markersize=8,
          label="New Sample",
        )


        ax.set_ylabel(r"$\mathrm{Y}$")
        plt.legend()
        plt.tight_layout()
        plt.show()