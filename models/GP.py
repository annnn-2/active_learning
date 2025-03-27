from .surrogate_model import SurrogateModel

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import copy

class GP_model(SurrogateModel):

    model_name: "GP"
    
    def __init__(self,
                 d: int,
                 kernel=None,
                 alpha: float = 1e-5,
                 n_restarts_optimizer: int = 5,
                 verbose: bool = False,
                 seed: int = 32):
        
        np.random.seed(seed)
        
        self.verbose = verbose
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        
        # Default kernel: RBF + ConstantKernel
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(d))
        
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=seed,
        )
        
        self.init_model = copy.deepcopy(self.gpr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, first: bool = False):
        
        self.gpr.fit(X, y)
        
        if first:
            self.init_model = copy.deepcopy(self.gpr)
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        
        if return_std:
            y_pred, y_std = self.gpr.predict(X, return_std=True)
            return y_pred, y_std
        else:
            y_pred = self.gpr.predict(X, return_std=False)
            return y_pred
            
    def params(self):
        
        return np.array([self.gpr.kernel_.k1.constant_value,self.gpr.kernel_.k2.length_scale,self.gpr.alpha_[0][0],])
    
    def Reset(self):
        
        self.gpr = copy.deepcopy(self.init_model)