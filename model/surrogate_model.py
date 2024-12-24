from abc import ABC, abstractmethod
import numpy as np


class SurrogateModel(ABC):

    model_name: str

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Placeholder for fit method

        Parameters
        ----------
        X: shape=(n_samples, n_features)
            input for training the surrogate model
        y: shape=(n_samples,1) 

        Returns
        -------
        model: Surrogate fitted model
        state: np.array model's parameters
        """
        return self

    @abstractmethod
    def predict(self,X):
        """
        Placeholder for prediction method

        Returns
        -------
        y: prediction
        """
        pass
        
    @abstractmethod
    def Reset(self,X):
        """
        Placeholder for reset method

        Returns
        -------
        state: np.array model's parameters
        """
        pass
   