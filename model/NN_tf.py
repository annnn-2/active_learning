import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation,BatchNormalization, Dropout, Input
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import tensorflow as tf
import tf_keras

import copy

from torchsummary import summary

from .surrogate_model import SurrogateModel

class NN_tf_model(SurrogateModel):
    
  model_name: "NN_tf"

  def __init__(self,
               d : int,
               nb_nodes: int = 40,
               nb_layers: int = 4,
               verbose: bool = False,
               n_epoches: int = 100,
               batch_size: int = 16,
               seed: int = 32,
              ):
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    self.verbose = verbose
    self.n_epoches = n_epoches
    self.batch_size = batch_size
      
    standard_opt = tf_keras.optimizers.legacy.Adam(learning_rate= 1e-3,
                name= 'Adam')   

    all_layers = []
    inputs = tf_keras.Input(shape=(d,), name='input_u')
       
    inp = inputs
    for layer in range(nb_layers + 1):
        layer_name = 'dense_u' + str(layer)
        this_layer = tf_keras.layers.Dense(nb_nodes, 
                                                activation='relu',
                                                name=layer_name,
                                                dtype=tf.float32)(inp)
        inp = this_layer
        all_layers.append(this_layer)
           
        
       
    output_layer = tf_keras.layers.Dense(1,
                                             activation= tf.keras.activations.linear,
                                             trainable=True,
                                             name='output_u',
                                             dtype=tf.float32)(inp)

   
    mse = tf_keras.losses.MeanSquaredError()
    self.approximation_nn = tf_keras.Model(inputs=inputs, outputs=output_layer) 
    self.approximation_nn.compile(loss=mse, optimizer=standard_opt, metrics=['accuracy'])
    self.init_model = copy.deepcopy(self.approximation_nn)
    self.init_params = list(self.init_model.get_weights())


  def fit(self,
          X: np.ndarray,
          y: np.ndarray,
          first: bool = False):
    self.approximation_nn.fit(X, y, batch_size= self.batch_size, 
                verbose=0, shuffle= True, epochs= self.n_epoches)

    if first:
      self.init_model = copy.deepcopy(self.approximation_nn)
      self.init_params = list(self.init_model.get_weights())

  def predict(self,X):   
    y_pred = self.approximation_nn(X)
    return y_pred.numpy().flatten()

  def params(self):     
    weights_list = self.approximation_nn.get_weights()
    flattened_weights = np.concatenate([w.flatten() for w in weights_list])
    return flattened_weights

  def Reset(self):
    self.approximation_nn = copy.deepcopy(self.init_model)
