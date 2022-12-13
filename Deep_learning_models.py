import tensorflow as tf
import pickle
import numpy as np


class DiffLearning_old(tf.keras.Model):

  def __init__(self, num_hidden_layers, num_cells_hidden_layer, activation, **kargs):

    '''
    Implements differential learning algorithm.
    Inputs:
    -------
    num_hidden_layers (int): number of hidden layers
    num_cells_hidden_layer (int): number of cells of a hidden layer.
    activation ('relu', 'elu'): activation of hidden layers.
    '''

    super().__init__(**kargs)

    self.num_hidden_layers = num_hidden_layers 
    self.model_layers = []
    self.activation = activation

    # We create the model layers as linear in order to have access 
    # to linear combination of previous layers.
    for i in range(num_hidden_layers):

      self.model_layers += [tf.keras.layers.Dense(units = num_cells_hidden_layer, activation = 'linear')] #, dtype = 'float32')]
 
    self.model_layers += [tf.keras.layers.Dense(units = 1, activation = 'linear')] #, dtype = 'float32')] 

  def elu_prime(self, data):
    '''
    Computes the derivative of elu(data) wrt data
    Inputs:
    -------
    data (tf.tensor)
    '''

    greater_than_0 = tf.experimental.numpy.heaviside(data, 0.0)
    return greater_than_0 + (1.0 - greater_than_0) * tf.exp(data)

  def relu_prime(self, data):
    '''
    Computes the derivative of relu(data) wrt data
    Inputs:
    -------
    data (tf.tensor)
    '''

    return tf.experimental.numpy.heaviside(data, 0.0)

  def call(self, X):
    '''
    Function called during fit and predict.
    Inputs:
    -------
    X (tf.tensor)
    '''

    # We build the computational graph from X to y
    z = []

    z += [self.model_layers[0](X)]

    if (self.num_hidden_layers > 0):

      for i in range(1, self.num_hidden_layers+1):

        if self.activation == 'elu':
          a = tf.keras.activations.elu(z[i-1])
        elif self.activation == 'relu':
          a = tf.keras.activations.relu(z[i-1])

        z += [self.model_layers[i](a)]

    # We build the computational graph of the adjoints dy/dy to dy/dx
    if (self.num_hidden_layers > 0):

      i = self.num_hidden_layers-1

      if self.activation == 'elu':
        adjoints = tf.transpose(self.model_layers[i+1].kernel) * self.elu_prime(z[i])
      elif self.activation == 'relu':
        adjoints = tf.transpose(self.model_layers[i+1].kernel) * self.relu_prime(z[i])

      for i in range(self.num_hidden_layers-2, -1, -1):

        if self.activation == 'elu':
          adjoints = tf.matmul(adjoints, tf.transpose(self.model_layers[i+1].kernel)) * self.elu_prime(z[i])
        elif self.activation == 'relu':
          adjoints = tf.matmul(adjoints, tf.transpose(self.model_layers[i+1].kernel)) * self.relu_prime(z[i])

      adjoints = tf.matmul(adjoints, tf.transpose(self.model_layers[0].kernel))

    else:

      adjoints = tf.transpose(self.model_layers[0].kernel)*tf.ones_like(z[-1])
    
    return tf.concat((z[-1],adjoints), axis = -1)

class DiffLearningLoss(tf.keras.losses.Loss):

  def __init__(self, alpha, deltas_L2_norm, **kwargs):

    super().__init__(**kwargs)
    self.alpha = alpha
    self.deltas_L2_norm = deltas_L2_norm

  def get_config(self):

    base_config = super().get_config()
    return {**base_config, 'alpha' : self.alpha, 'deltas_L2_norm': self.deltas_L2_norm}

  def call(self, y_true, y_pred):

    value_true = y_true[:,0]
    value_pred = y_pred[:,0]

    sens_true = y_true[:,1:]
    sens_pred = y_pred[:,1:]

    # return tf.reduce_mean((value_true - value_pred) ** 2) + self.alpha * tf.reduce_sum((sens_true - sens_pred) ** 2) / (y_true.shape[0] * (y_true.shape[1]-1))

    return tf.reduce_mean((value_true - value_pred) ** 2) + self.alpha * tf.reduce_mean(((sens_true - sens_pred) ** 2)/self.deltas_L2_norm) 

class Diff_learning_scaler:

  def __init__(self, diff_learning_model, alpha = 1.0):

    self.diff_learning_model = diff_learning_model
    self.alpha = alpha

  def normalize_data(self, X, y, dydX):

    self.X_mu = np.mean(X, axis = 0)
    self.X_sigma = np.std(X, axis = 0)

    self.y_mu = np.mean(y)
    self.y_sigma = np.std(y)

    X_scaled = (X - self.X_mu) / self.X_sigma 
    y_scaled = (y - self.y_mu) / self.y_sigma 
    
    dydX_scaled = dydX * self.X_sigma / self.y_sigma  

    self.dydX_scaled_L2_norm = np.mean(dydX_scaled*dydX_scaled, axis = 0)

    self.loss = DiffLearningLoss(self.alpha, self.dydX_scaled_L2_norm)

    return {'X_scaled': X_scaled, 'y_scaled': y_scaled, 'dydX_scaled': dydX_scaled}

  def fit(self, X, y, dydX, batch_size= 32, epochs= 20):
    
    scaled_data = self.normalize_data(X, y, dydX)

    self.diff_learning_model.compile(optimizer = 'adam', loss = self.loss)

    y_to_model = np.concatenate((scaled_data['y_scaled'].reshape(-1,1), scaled_data['dydX_scaled']), axis =1)  

    return self.diff_learning_model.fit(scaled_data['X_scaled'], y_to_model, batch_size, epochs, shuffle= False)

  def predict(self, X, batch_size = 32):

    X_scaled = (X - self.X_mu) / self.X_sigma 

    y_sens_scaled_predicted = self.diff_learning_model.predict(X_scaled, batch_size = batch_size)

    y = self.y_mu +  y_sens_scaled_predicted[:,0]*self.y_sigma

    sens = y_sens_scaled_predicted[:,1:] * self.y_sigma / self.X_sigma  

    return {'y': y, 'sens': sens}

  def save(self, PATH):

    self.diff_learning_model.save(PATH + 'model')

    params = {'alpha': self.alpha, 'X_mu': self.X_mu, 'X_sigma': self.X_sigma, 'y_mu': self.y_mu, 
              'y_sigma': self.y_sigma, 'dydX_scaled_L2_norm': self.dydX_scaled_L2_norm }

    with open(PATH + 'params.pkl', 'wb') as f:
      pickle.dump(params, f)

        
  def open(PATH):

    with open(PATH + 'params.pkl', 'rb') as f:
      params = pickle.load(f)

    loss = DiffLearningLoss(alpha= params['alpha'],deltas_L2_norm=  params['dydX_scaled_L2_norm'])
    
    diff_learning_model = tf.keras.models.load_model(PATH + 'model', 
        custom_objects= {'DiffLearningLoss': loss})

    model = Diff_learning_scaler(diff_learning_model, alpha= params['alpha'])

    model.alpha = params['alpha']
    model.X_mu = params['X_mu']
    model.X_sigma = params['X_sigma']
    model.y_mu = params['y_mu']
    model.y_sigma = params['y_sigma']
    model.dydX_scaled_L2_norm = params['dydX_scaled_L2_norm']

    return model

class DiffLearning(tf.keras.Model):

  def __init__(self, model, **kargs):

    '''
    Implements differential learning algorithm.
    Inputs:
    -------
    model (keras model) with a single output
    '''

    super().__init__(**kargs)

    self.model = model 
    
    
  def call(self, X):
    '''
    Function called during fit and predict.
    Inputs:
    -------
    X (tf.tensor)
    '''
    with tf.GradientTape() as t:

      t.watch(X)
      y = self.model(X)

    adjoints = t.gradient(y, X)

    
    return tf.concat((y,adjoints), axis = -1)


def build_dense_model(input_shape, num_hidden_layers, num_neurons_hidden_layers, hidden_layer_activation, output_layer_activation):
  '''
  Builds a tf dense model.
  Inputs:
  -------
  * input_shape (int): dimension of input
  * num_hidden_layers (int): number of hidden layers.
  * num_neurons_hidden_layers (int): number of neurons per hidden layer.
  * hidden_layer_activation (str of tf.keras.activation): activation function for hidden layers.
  * output_layer_activation (str of tf.keras.activation): activation function for output layers.
  '''

  dense_model = tf.keras.Sequential()

  if num_hidden_layers>1:

    dense_model.add(tf.keras.layers.Dense(num_neurons_hidden_layers, input_shape=(input_shape,), activation = hidden_layer_activation))
  
    for i in range(num_hidden_layers-1):
    
      dense_model.add(tf.keras.layers.Dense(num_neurons_hidden_layers, activation = hidden_layer_activation))
  
  dense_model.add(tf.keras.layers.Dense(1, activation = output_layer_activation))

  return dense_model


