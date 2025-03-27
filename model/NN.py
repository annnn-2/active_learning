import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from .surrogate_model import SurrogateModel

class NN_model(nn.Module,SurrogateModel):

    model_name: "NN"
    
    def __init__(self,
                 d: int,
                 nb_nodes: int = 40,
                 nb_layers: int = 4,
                 verbose: bool = False,
                 seed: int = 32,
                 n_epoches: int = 100,
                 batch_size: int = 16):
        super(NN_model, self).__init__()
        
        torch.manual_seed(seed)
        
        self.verbose = verbose
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d, nb_nodes))
        for _ in range(nb_layers):
            self.layers.append(nn.Linear(nb_nodes, nb_nodes))
        self.output_layer = nn.Linear(nb_nodes, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        self.criterion = nn.MSELoss()
       
        self.init_model = copy.deepcopy(self.state_dict())
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            first: bool = False):
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.n_epoches):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.n_epoches}, Loss: {loss.item()}')
        
        if first:
            self.init_model = copy.deepcopy(self.state_dict())
    
    def predict(self, X: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self(X)
        return y_pred.numpy().flatten()
        
    def params(self):
        params = self.state_dict()
        flat_params = []
        
        for i in params:
            flat_params.append(np.array(params[i]).flatten())
        return np.concatenate(flat_params)
    
    def Reset(self):
        self.load_state_dict(self.init_model)