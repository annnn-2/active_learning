from abc import ABC, abstractmethod

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
    def params(self):
        """
        Placeholder for parameters method

        Returns
        -------
        parameters: np.array model's flatten parameters
        """
        pass
        
    @abstractmethod
    def Reset(self):
        """
        Placeholder for reset method

        Returns
        -------

        """
        pass
   