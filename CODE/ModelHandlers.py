import Miscellanea
from itertools import product
import numpy as np
import Deep_learning_models
import tensorflow as tf
import os
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.stats import ks_2samp 


def TrainSetOfModels(PATH_MODELS, alphas, cells_layer, num_hidden_layers, sim_scenarios_train, payoff_train, sens_train, epochs ,batch_size, valid_data = None):

    '''
    Trains a set of deep learning models.
    Inputs:
    _______
    * PATH_MODELS (str): Path where to save the set of models.
    * alphas (list(float)): Contains set of alpha (diff machine learning parameter).
    * cells_layer (list(int)): Contains set of number of cells per layer.
    * num_hidden_layers (list(int)): Contains set of number of hidden layers.
    * sim_scenarios_train (np.array) : Contains X variable (number of scenarios x number of risk factors).
    * payoff_train (np.array): Contains y variable (number of scenarios)
    * sens_train (np.array): Contains pathwise differentials (umber of scenarios x number of risk factors)
    * epochs: number of epochs
    * batch_size: batch size
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
        dense_model = Deep_learning_models.build_dense_model(sim_scenarios_train.shape[1],n,c,'relu','linear')

        # Wrap the model in a "Diff_learning_scaler" class, which applies a scaling factor alpha to the gradients during training
        model = Deep_learning_models.Diff_learning_scaler(Deep_learning_models.DiffLearning(dense_model), alpha = a)

         # Train the model using the fit method with the given batch size and number of epochs, and save the resulting model in the previously created directory
        model.fit(sim_scenarios_train, payoff_train, sens_train, batch_size= batch_size, 
                epochs= epochs, validation_data = valid_data)

        model.save(path)
        
        clear_output()





def load_set_of_models(PATH_MODELS, alphas, cells_layer, num_hidden_layer, sim_scenarios_train, payoff_train, sim_scenarios_cv, payoff_cv, closed_formula_train, closed_formula_train_cv):
    """
    Loads a set of models and computes various evaluation metrics for each model.
    
    Args:
    _____
    * PATH_MODELS (str): The path to the directory where the models are stored.
    * alphas (list of float): A list of values for the alpha hyperparameter.
    * cells_layer (list of int): A list of values for the number of cells in each layer of the neural network.
    * num_hidden_layer (list of int): A list of values for the number of hidden layers in the neural network.
    * sim_scenarios_train (numpy.ndarray): An array of simulated scenarios used for training the models.
    * payoff_train (numpy.ndarray): An array of payoff values corresponding to the simulated scenarios used for training.
    * sim_scenarios_cv (numpy.ndarray): An array of simulated scenarios used for cross-validation.
    * payoff_cv (numpy.ndarray): An array of payoff values corresponding to the simulated scenarios used for cross-validation.
    * closed_formula_train (numpy.ndarray): An array of closed-formula prices corresponding to the simulated scenarios used for training.
    * closed_formula_train_cv (numpy.ndarray): An array of closed-formula prices corresponding to the simulated scenarios used for cross-validation.
    
    Returns:
    ________
    dict: A dictionary where the keys are tuples of (alpha, num_hidden_layer, cells_layer) and the values are dictionaries with various evaluation metrics for each model.
    """
    models = {}

    # Iterate over all combinations of hyperparameters
    for a, c, n in product(alphas, cells_layer, num_hidden_layer):

        # Construct the name of the model based on the hyperparameters
        model_name = 'alpha' + '_' + str(a) + '_cells_' + str(c) + '_hidden_' + str(n) 

        # Print the name of the current model
        print(model_name)

        # Construct the path to the directory where the current model is stored
        path = PATH_MODELS +  model_name + '/'

        # Load the current model
        model = Deep_learning_models.Diff_learning_scaler.open(path)

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the training data
        y_pred_train = model.predict(sim_scenarios_train, batch_size = len(sim_scenarios_train))['y']
        mse_train = mean_squared_error(y_pred_train, payoff_train)

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the cross-validation data
        y_pred_cv = model.predict(sim_scenarios_cv, batch_size = len(sim_scenarios_cv))['y']
        mse_cv = mean_squared_error(y_pred_cv, payoff_cv)

        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the training data
        spearman_train = spearmanr(y_pred_train, closed_formula_train)
        ks_train = ks_2samp(y_pred_train, closed_formula_train)

        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the cross-validation data
        spearman_cv = spearmanr(y_pred_cv, closed_formula_train_cv)
        ks_cv = ks_2samp(y_pred_cv, closed_formula_train_cv)

        # Construct a dictionary containing the results for the current model
        results_dict = {}
        results_dict['model_name'] = model_name
        results_dict['model_params'] = {'alpha': a, 'cells': c, 'hidden': n}

        results_dict['model'] = model
        results_dict['mse_train'] = mse_train
        results_dict['mse_cv'] = mse_cv
        results_dict['spearman_train'] = spearman_train
        results_dict['spearman_cv'] = spearman_cv
        results_dict['ks_train'] = ks_train
        results_dict['ks_cv'] = ks_cv

        # Add the results for the current model to the dictionary of all models
        models[(a,n,c)] = results_dict

        # Clear the output to keep the notebook clean
        clear_output()

    return models

