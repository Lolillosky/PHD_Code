import Miscellanea
from itertools import product
import numpy as np
import Deep_learning_models
import tensorflow as tf
import os
from IPython.display import clear_output


def TrainSetOfModels(PATH_MODELS, alphas, cells_layer, num_hidden_layers, sim_scenarios_train, payoff_train, sens_train, epochs ,batch_size, valid_data = None):

    '''
    Trains a set of deep learning models.
    Inputs:
    -------
    * PATH_MODELS (str): Path where to save the set of models.
    * alphas (list(float)): Contains set of alpha (diff machine learning parameter).
    * cells_layer (list(int)): Contains set of number of cells per layer.
    * num_hidden_layers (list(int)): Contains set of number of hidden layers.
    * sim_scenarios_train (np.array) : Contains X variable (number of scenarios x number of risk factors).
    * payoff_train (np.array): Contains y variable (number of scenarios)
    * sens_train (np.array): Contains pathwise differentials (umber of scenarios x number of risk factors)
    * valid_data (dict): contains the following:
        * valid_data['X_val'] = cv X
        * valid_data['y_val'] = cv y
        * valid_data['dydX_val'] = cv sens
        * valid_data['patience'] = Early stopping patience
    '''

     # Delete the contents of the folder where the trained models will be saved
    Miscellanea.delete_content_of_folder(PATH_MODELS)


    # Iterate over different values of hyperparameters
    for a, c, n in product(alphas, cells_layer, num_hidden_layers):

        # Set a random seed to ensure reproducibility of results
        tf.keras.utils.set_random_seed(9876)

         # Create a new directory to save the trained model and print a message indicating the current hyperparameter values
        print('FITTING MODEL: alpha: {0}, cells: {1}, num_hidden: {2}'.format(a, c, n))

        path = PATH_MODELS + 'alpha' + '_' + str(a) + '_cells_' + str(c) + '_hidden_' + str(n) + '/'
        os.mkdir(path)

         # Build a deep learning model with the specified number of hidden layers and cells per layer using the "build_dense_model" function
        dense_model = Deep_learning_models.build_dense_model(sim_scenarios_train.shape[1],n,c,'softplus','linear')

        # Wrap the model in a "Diff_learning_scaler" class, which applies a scaling factor alpha to the gradients during training
        model = Deep_learning_models.Diff_learning_scaler(Deep_learning_models.DiffLearning(dense_model), alpha = a)

         # Train the model using the fit method with the given batch size and number of epochs, and save the resulting model in the previously created directory
        model.fit(sim_scenarios_train, payoff_train, sens_train, batch_size= batch_size, 
                epochs= epochs, validation_data = valid_data)

        model.save(path)
        
        clear_output()
