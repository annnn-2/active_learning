import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary
import copy

from torchsummary import summary

from .surrogate_model import SurrogateModel

import matplotlib.pyplot as plt

class NN_model(SurrogateModel):
    
  model_name: "NN"

  def __init__(self,
               d : int,
               nb_nodes: int = 40,
               nb_layers: int = 4,
               verbose: bool = False):

    self.approximation_nn = nn.Sequential()
    self.approximation_nn.add_module('layer1', nn.Linear(d,nb_nodes))
    self.approximation_nn.add_module('elu1', nn.ELU())
    for l in range(nb_layers-2):
      self.approximation_nn.add_module('layer'+str(l+2), nn.Linear(nb_nodes,nb_nodes))
      self.approximation_nn.add_module('relu'+str(l+2), nn.ELU())

    self.approximation_nn.add_module('layer'+str(nb_layers), nn.Linear(nb_nodes,1))

    self.verbose = verbose
      
    if self.verbose:
        summary(self.approximation_nn,(d,))

    self.loss_fn = nn.MSELoss(reduction="mean")
    self.optimizer = torch.optim.Adam(self.approximation_nn.parameters(), lr=0.00001)#, momentum=0.9)

    #initial model and its parameters if no initial training set is provided
    self.init_model = copy.deepcopy(self.approximation_nn)
    self.init_params = list(self.init_model.parameters())

  def fit(self,
          X: np.ndarray,
          y: np.ndarray,
          n_epochs: int = 1000,
          batch_size: int = 10,
          first: bool = False):

    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)

    loss_values =[]

    for epoch in range(n_epochs):
      running_loss = 0.0
      for i in range(0, len(X), batch_size):
          Xbatch = X[i:i+batch_size]
          y_pred = self.approximation_nn(Xbatch)
          ybatch = y[i:i+batch_size]
          loss = self.loss_fn(y_pred, ybatch)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          running_loss =+ loss.item()
      if epoch > 0 and np.abs(loss_values[-1] - running_loss) < 0.0001:
        break
      loss_values.append(running_loss / len(X))

    if self.verbose:
        print('loss values')
        plt.plot(loss_values)

    self.loss = loss.item()

    if first:
      self.init_model = copy.deepcopy(self.approximation_nn)
      self.init_params = list(self.init_model.parameters())

    state = np.array([])
    wb = list(self.approximation_nn.parameters())
    for i in range(0, len(wb)):
        vec = wb[i].data.flatten().numpy()
        state = np.concatenate((state , vec))

    return self.approximation_nn, state

  def predict(self,X):
      
    X = torch.tensor(X,dtype=torch.float32)
    with torch.no_grad():
      y_pred = self.approximation_nn(X)
    return y_pred

  def Reset(self):

    state = np.array([])
    mp = list(self.approximation_nn.parameters())
    n = len(self.init_params)
    for i in range(0, n):
        mp[i].data[:] = self.init_params[i].data[:]

        vec = self.init_params[i].data.flatten().numpy()
        state = np.concatenate((state , vec))

    return state

  # def Reset(self):

  #   state = np.array([])
  #   mp = list(self.approximation_nn.parameters())
  #   n = len(self.init_params)
  #   for i in range(0, n):
  #       mp[i].data[:] = self.init_params[i].data[:]

  #       vec = self.init_params[i].data.flatten().numpy()
  #       state = np.concatenate((state , vec))

  #   return state