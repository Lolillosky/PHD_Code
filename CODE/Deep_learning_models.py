import tensorflow as tf
import pickle
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import ks_2samp 
import Miscellanea

import Enums


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

  """ def __init__(self, alpha, deltas_L2_norm, name = 'DiffLearningLoss'):

    super().__init__(name = name)
    self.alpha = alpha
    self.deltas_L2_norm = deltas_L2_norm
 """
  def __init__(self, alpha, deltas_L2_norm, **kwargs):

    super().__init__(**kwargs)
    self.alpha = alpha
    self.deltas_L2_norm = deltas_L2_norm

  def get_config(self):

    base_config = super().get_config()
    base_config.update({'alpha': self.alpha, 'deltas_L2_norm': self.deltas_L2_norm})
    return base_config


  @classmethod
  def from_config(cls, config):
      return cls(**config)

  @tf.function
  def call(self, y_true, y_pred):

    #y_true = tf.convert_to_tensor(y_true,  dtype=tf.float64)
    #y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float64)

    value_true = y_true[:,0]
    value_pred = y_pred[:,0]

    sens_true = y_true[:,1:]
    sens_pred = y_pred[:,1:]

    # TODO: Add small number to self.deltas_L2_norm
    return tf.reduce_mean(tf.square(value_true - value_pred)) + self.alpha * tf.reduce_mean((tf.square(sens_true - sens_pred))/(self.deltas_L2_norm)) 


class DiffLearningEarlySpotLoss(tf.keras.metrics.Metric):


  def __init__(self, y_mu, y_sigma, **kwargs):

    super().__init__(**kwargs)
    self.y_mu = y_mu
    self.y_sigma = y_sigma

    self.mean = self.add_weight(name='mean',initializer='zeros')
    self.count = self.add_weight(name='count',initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    value_true = y_true[:,0]
    value_pred = y_pred[:,0]

    value_true = self.y_mu + self.y_sigma * value_true
    value_pred = self.y_mu + self.y_sigma * value_pred
    
    self.mean.assign_add(tf.math.reduce_sum(tf.math.square(value_pred - value_true)))
    self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.keras.backend.floatx()))

  def result(self):
    return self.mean / self.count
    
  def reset_state(self):
    self.mean.assign(0.)
    self.count.assign(0.)

  def get_config(self):

    base_config = super().get_config()
    base_config.update({'y_mu': self.y_mu, 'y_sigma': self.y_sigma})

    return base_config

  @classmethod
  def from_config(cls, config):
      return cls(**config)
  
class EpochCounterCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Increment the number of epochs parameter of the model
        self.model.num_epochs += 1

  
class Diff_learning_scaler:

  def __init__(self, diff_learning_model, alpha = 1.0):

    self.diff_learning_model = diff_learning_model
    self.alpha = alpha

  def compute_normalization_params(self, X, y, dydX):

    self.X_mu = np.mean(X, axis = 0)
    self.X_sigma = np.std(X, axis = 0)

    self.y_mu = np.mean(y)
    self.y_sigma = np.std(y)

    dydX_scaled = dydX * self.X_sigma / self.y_sigma  

    self.dydX_scaled_L2_norm = np.mean(dydX_scaled*dydX_scaled, axis = 0)


  def normalize_data(self, X, y, dydX):

    
    X_scaled = (X - self.X_mu) / self.X_sigma 
    y_scaled = (y - self.y_mu) / self.y_sigma 
    
    dydX_scaled = dydX * self.X_sigma / self.y_sigma  

    return {'X_scaled': X_scaled, 'y_scaled': y_scaled, 'dydX_scaled': dydX_scaled}

  def fit(self, X, y, dydX, batch_size= 32, epochs= 20, validation_data = None, plat_callback = None, verbose = 0):

    '''
    Inputs:
    -------
    validation_data: dict(keys = {X_val, y_val, dydX_val, patience})
    '''

    self.compute_normalization_params(X, y, dydX)

    self.loss = DiffLearningLoss(self.alpha, self.dydX_scaled_L2_norm)

    scaled_data = self.normalize_data(X, y, dydX)

    y_to_model = np.concatenate((scaled_data['y_scaled'].reshape(-1,1), scaled_data['dydX_scaled']), axis =1)  

    epoch_counter = EpochCounterCallback()
    
    if validation_data is not None:

      X_val = validation_data['X_val']
      y_val = validation_data['y_val']
      dydX_val = validation_data['dydX_val']
      
      scaled_data_val = self.normalize_data(X_val, y_val, dydX_val)
      y_to_model_val = np.concatenate((scaled_data_val['y_scaled'].reshape(-1,1), scaled_data_val['dydX_scaled']), axis =1)  

      loss_val = DiffLearningEarlySpotLoss(y_mu = self.y_mu, y_sigma = self.y_sigma, name = 'Myval_Loss')

      early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_Myval_Loss", patience=validation_data['patience'], start_from_epoch = validation_data['start_from_epoch'], 
                restore_best_weights=True, mode="min")

  
      self.diff_learning_model.compile(optimizer = 'adam', loss = self.loss, metrics = [loss_val])

      callbacks=[early_stop, epoch_counter]

      if plat_callback is not None:

        callbacks += [plat_callback]

      
      return self.diff_learning_model.fit(scaled_data['X_scaled'], y_to_model, batch_size, epochs, shuffle= False, callbacks=callbacks,
                validation_data = (scaled_data_val['X_scaled'], y_to_model_val), verbose = verbose)

    else:

      callbacks=[epoch_counter]

      if plat_callback is not None:

        callbacks += [plat_callback]

      self.diff_learning_model.compile(optimizer = 'adam', loss = self.loss)

      return self.diff_learning_model.fit(scaled_data['X_scaled'], y_to_model, batch_size, epochs, shuffle= False, callbacks=callbacks, 
                                           verbose = verbose)

  def predict(self, X, batch_size = 32, verbose = 0):

    X_scaled = (X - self.X_mu) / self.X_sigma 

    y_sens_scaled_predicted = self.diff_learning_model.predict(X_scaled, batch_size = batch_size, verbose = verbose)

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

    #diff_learning_model = tf.keras.models.load_model(PATH + 'model', 
    #    custom_objects= {'DiffLearningLoss': loss})

    diff_learning_model = tf.keras.models.load_model(PATH + 'model', 
        custom_objects= {'DiffLearningLoss': DiffLearningLoss(alpha= params['alpha'],deltas_L2_norm=  params['dydX_scaled_L2_norm']), 
        'DiffLearningEarlySpotLoss': DiffLearningEarlySpotLoss(y_mu=params['y_mu'], y_sigma = params['y_sigma'])})
    
    #diff_learning_model = tf.keras.models.load_model(PATH + 'model',compile=False)

    model = Diff_learning_scaler(diff_learning_model, alpha= params['alpha'])

    model.alpha = params['alpha']
    model.X_mu = params['X_mu']
    model.X_sigma = params['X_sigma']
    model.y_mu = params['y_mu']
    model.y_sigma = params['y_sigma']
    model.dydX_scaled_L2_norm = params['dydX_scaled_L2_norm']

    return model

class DiffLearning(tf.keras.Model):

  def __init__(self, model, num_epochs=0, **kargs):

    '''
    Implements differential learning algorithm.
    Inputs:
    -------
    model (keras model) with a single output
    '''

    super().__init__(**kargs)

    self.model = model 
    self._num_epochs = num_epochs
    
    
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
  
  def get_config(self):
        # Add the num_epochs parameter to the model configuration
        config = super().get_config()
        config.update({'num_epochs': self._num_epochs})
        return config

  @classmethod
  def from_config(cls, config):
    # Create a new model instance from the given configuration
    return cls(num_epochs=config['num_epochs'])
  
  @property
  def num_epochs(self):
    # Define a getter method to retrieve the num_epochs value from the configuration dictionary
    return self._num_epochs

  @num_epochs.setter
  def num_epochs(self, value):
    # Define a setter method to set the num_epochs value
    self._num_epochs = value
  

def build_dense_model(input_shape, num_hidden_layers, num_neurons_hidden_layers, hidden_layer_activation, output_layer_activation):

  dense_model = tf.keras.Sequential()

  if num_hidden_layers>=1:

    dense_model.add(tf.keras.layers.Dense(num_neurons_hidden_layers, input_shape=(input_shape,), activation = hidden_layer_activation))
  
    for i in range(num_hidden_layers-1):
    
      dense_model.add(tf.keras.layers.Dense(num_neurons_hidden_layers, activation = hidden_layer_activation))
  
  dense_model.add(tf.keras.layers.Dense(1, activation = output_layer_activation))

  return dense_model

def build_diff_learning_model(input_shape, num_hidden_layers, num_neurons_hidden_layers, hidden_layer_activation,
                output_layer_activation, alpha):
  
  dense_model = build_dense_model(input_shape, num_hidden_layers, num_neurons_hidden_layers, hidden_layer_activation, output_layer_activation)

  diff_learning_model = DiffLearning(dense_model)

  return Diff_learning_scaler(diff_learning_model, alpha)
  



class Plat_Callback(tf.keras.callbacks.Callback):
    def __init__(self, base_scenario_dict, test_scenario_dict, diff_learning_scaler, base_scenario_adj_option, count_show):

      '''
      - 'scenario' (numpy.ndarray): An array of simulated scenarios used for test.
        - 'closed_formula_plus_adj' (numpy.ndarray): An array of the closed form formula plus adjustments. See comments above.
        - 'model_adj': array of model adjustments not dependent on base scenario. See comments above.
        - 'base_scenario_closed_form_sens': only yo be included for base scenario.
      '''
      self.base_scenario_dict = Miscellanea.deep_copy_dict_with_arrays(base_scenario_dict)
      self.test_scenario_dict = Miscellanea.deep_copy_dict_with_arrays(test_scenario_dict)


      self.diff_learning_scaler = diff_learning_scaler
      self.base_scenario_adj_option = base_scenario_adj_option

      self.ks_stat = []
      self.rank_corr = []
      self.count_show = count_show
      self.batch_count = 0 
      self.output_dict = {}

      self.output_dict['ks_stat'] = []
      self.output_dict['rank_corr'] = []
      self.output_dict['y_true'] = []
      self.output_dict['y_pred'] = []
      self.output_dict['batch_count'] = []
      
      
  
      

    def on_batch_end(self, epoch, logs=None):
        
        if (self.batch_count % self.count_show) == 0:

          model_predict_test_y = self.diff_learning_scaler.predict(self.test_scenario_dict['scenario'], batch_size = len(self.test_scenario_dict['scenario']))['y']

          
          if  (self.base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV) or \
                (self.base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV_PLUS_SENS):
            
            base_scen_predict = self.diff_learning_scaler.predict(self.base_scenario_dict['scenario'], batch_size = 1)

            base_scen_y = base_scen_predict['y']
            
            base_scen_sens = base_scen_predict['sens']

            model_adj_base = -base_scen_y
          
          else:

            model_adj_base = 0.0
            
            

          if (self.base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV_PLUS_SENS):
           
              model_adj_base_sens_test = np.matmul(self.test_scenario_dict['scenario']-self.base_scenario_dict['scenario'],
                                                  (self.base_scenario_dict['base_scenario_closed_form_sens'] - base_scen_sens).T).flatten()
          else:
              model_adj_base_sens_test = 0.0

          y_true = self.test_scenario_dict['closed_formula_plus_adj']
          y_pred = model_predict_test_y + model_adj_base + model_adj_base_sens_test + self.test_scenario_dict['model_adj']
          # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for historical scenarios
          spearman_test, _ = spearmanr(y_pred, y_true)
          ks_hist_test, _ = ks_2samp(y_pred, y_true)

          self.output_dict['ks_stat'] += [ks_hist_test]
          self.output_dict['rank_corr'] += [spearman_test]

          self.output_dict['y_true'] += [y_true]
          self.output_dict['y_pred'] += [y_pred]
          self.output_dict['batch_count'] += [self.batch_count + 1]
  

          # Miscellanea.plot_plat_charts(model_predict_test_y + model_adj_base + model_adj_base_sens_test + self.test_scenario_dict['model_adj'], self.test_scenario_dict['closed_formula_plus_adj'])
        
        self.batch_count += 1
