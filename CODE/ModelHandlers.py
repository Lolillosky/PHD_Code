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
import matplotlib.pyplot as plt


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


def load_models(PATH_MODELS, alphas, cells_layer, num_hidden_layers):
    """
    Load deep learning models from the specified directory based on different hyperparameters.

    Parameters:
    ----------
    PATH_MODELS : str
        The path to the directory containing the saved models.
    alphas : list
        A list of learning rates for the deep learning models.
    cells_layer : list
        A list of the number of cells in each layer of the deep learning models.
    num_hidden_layers : list
        A list of the number of hidden layers in the deep learning models.

    Returns:
    -------
    dict
        A dictionary of the loaded models, where each key is a tuple of hyperparameters (alpha, num_hidden_layers, cells_layer) and each value is a dictionary containing the name of the model and its hyperparameters.
    """

    # Initialize an empty dictionary to store the loaded models
    models = {}

    # Loop over all combinations of the hyperparameters
    for a, c, n in product(alphas, cells_layer, num_hidden_layers):

        # Construct the name of the model based on the hyperparameters
        model_name = 'alpha' + '_' + str(a) + '_cells_' + str(c) + '_hidden_' + str(n)

        # Print the name of the current model
        print(model_name)

        # Construct the path to the directory where the current model is stored
        path = PATH_MODELS + model_name + '/'

        # Load the current model using a function named "open" from the "Deep_learning_models.Diff_learning_scaler" module
        model = Deep_learning_models.Diff_learning_scaler.open(path)

        # Create a dictionary to store the results of each loaded model
        results_dict = {}

        # Add the name of the current model to the dictionary
        results_dict['model_name'] = model_name

        # Add the hyperparameters of the current model to the dictionary
        results_dict['model_params'] = {'alpha': a, 'cells': c, 'hidden': n}

        results_dict['model'] = model


        # Store the dictionary for the current model in the models dictionary using the hyperparameters as keys
        models[(a,n,c)] = results_dict

        # Clear the output to keep the notebook clean
        clear_output()

    # Return the dictionary of loaded models
    return models

def compute_model_metrics(models, alphas, cells_layer, num_hidden_layers, sim_scenarios_train, payoff_train, sim_scenarios_cv, payoff_cv, 
                       closed_formula_plus_adj_train, closed_formula_plus_adj_cv, model_adj_train, model_adj_cv,
                       test_scenarios_data, adjust_model_with_base_scenario):
    """
    Compute metrics for multiple models with different hyperparameters.

    Parameters:
    models (dict): dictionary of trained models with hyperparameters as keys.
    alphas (list): list of alpha values to use in the models.
    cells_layer (list): list of the number of cells in each LSTM layer to use in the models.
    num_hidden_layers (list): list of the number of hidden layers to use in the models.
    sim_scenarios_train (np.ndarray): array of simulated scenarios for training.
    payoff_train (np.ndarray): array of actual payoffs for training.
    sim_scenarios_cv (np.ndarray): array of simulated scenarios for cross-validation.
    payoff_cv (np.ndarray): array of actual payoffs for cross-validation.
    closed_formula_plus_adj_train (np.ndarray): array of closed formula prices plus adjustments for training.
    closed_formula_plus_adj_cv (np.ndarray): array of closed formula prices plus adjustments for cross-validation.
    model_adj_train (np.ndarray): array of model adjustments for training.
    model_adj_cv (np.ndarray): array of model adjustments for cross-validation.
    test_scenarios_data (list): list of dictionaries containing test scenarios with keys for scenario name, scenario data, 
                                closed formula prices plus adjustments, and model adjustments. Keys:
        - 'scenario_name' (str)
        - 'scenario' (numpy.ndarray): An array of simulated scenarios used for test.
        - 'closed_formula_plus_adj' (numpy.ndarray): An array of the closed form formula plus adjustments.
        - 'model_adj': array of model adjustments not dependent on base scenario.
    adjust_model_with_base_scenario (bool): flag indicating whether to adjust model with base scenario.

    Returns:
    dict: dictionary of model metrics with hyperparameters as keys and metrics as values.
    """
    metrics = {}

    # Iterate over all combinations of hyperparameters
    for a, c, n in product(alphas, cells_layer, num_hidden_layers):

        # Construct the name of the model based on the hyperparameters
        model_name = 'alpha' + '_' + str(a) + '_cells_' + str(c) + '_hidden_' + str(n) 

        # Print the name of the current model
        print(model_name)

        # Load the current model
        model = models[(a,n,c)]['model']

        if adjust_model_with_base_scenario:
            # We calculate the estimate for the base scenario
            model_adj_base = - model.predict(test_scenarios_data[0]['scenario'], batch_size = 1)['y']
        else:
            model_adj_base = 0.0

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the training data
        y_pred_train = model.predict(sim_scenarios_train, batch_size = len(sim_scenarios_train))['y']
        mse_train = mean_squared_error(y_pred_train, payoff_train)

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the cross-validation data
        y_pred_cv = model.predict(sim_scenarios_cv, batch_size = len(sim_scenarios_cv))['y']
        mse_cv = mean_squared_error(y_pred_cv, payoff_cv)

        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the training data
        spearman_train, _ = spearmanr(y_pred_train + model_adj_base + model_adj_train, closed_formula_plus_adj_train)
        ks_train, _ = ks_2samp(y_pred_train + model_adj_base + model_adj_train, closed_formula_plus_adj_train)


        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the cross-validation data
        spearman_cv, _ = spearmanr(y_pred_cv + model_adj_base + model_adj_cv, closed_formula_plus_adj_cv)
        ks_cv, _ = ks_2samp(y_pred_cv + model_adj_base + model_adj_cv, closed_formula_plus_adj_cv)

        # Construct a dictionary containing the results for the current model
        results_dict = {}
        results_dict['model_name'] = model_name
        results_dict['model_params'] = {'alpha': a, 'cells': c, 'hidden': n}
        #results_dict['model'] = model
        results_dict['mse_train'] = mse_train
        results_dict['mse_cv'] = mse_cv
        results_dict['spearman_train'] = spearman_train
        results_dict['spearman_cv'] = spearman_cv
        results_dict['ks_train'] = ks_train
        results_dict['ks_cv'] = ks_cv
        
        # Add the results for the current model to the dictionary of all models
        metrics[(a,n,c)] = results_dict
        
        for i in range(1,len(test_scenarios_data)):
            # Predict model for historical scenarios
            y_pred_test = model.predict(test_scenarios_data[i]['scenario'], batch_size = len(test_scenarios_data[i]['scenario']))['y']

            # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for historical scenarios
            spearman_test, _ = spearmanr(y_pred_test + model_adj_base + test_scenarios_data[i]['model_adj'], test_scenarios_data[i]['closed_formula_plus_adj'])
            ks_hist_test, _ = ks_2samp(y_pred_test + model_adj_base + test_scenarios_data[i]['model_adj'], test_scenarios_data[i]['closed_formula_plus_adj'])

            results_dict['spearman_' + test_scenarios_data[i]['scenario_name']] = spearman_test
            results_dict['ks_' + test_scenarios_data[i]['scenario_name']] = ks_hist_test

        # Clear the output to keep the notebook clean
        clear_output()
    
    mse_list = []
    mse_keys = []

    for key in metrics:

        mse_keys += [key]
        mse_list += [metrics[key]['mse_cv']]   


    print('Best model:')
    print(metrics[mse_keys[np.argmin(mse_list)]])

    return metrics




def load_set_of_models_and_compute_metrics(PATH_MODELS, alphas, cells_layer, num_hidden_layers, sim_scenarios_train, payoff_train, sim_scenarios_cv, payoff_cv, 
                       closed_formula_train, closed_formula_train_cv, test_scenarios_and_closed_formula):
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
    * test_scenarios_and_closed_formula list(dict): list of dictionaries containing historical scenarios, their closed form formulas and their names. First element is base scenario. Keys:
        - 'scenario_name' (str)
        - 'scenario' (numpy.ndarray): An array of simulated scenarios used for cross-validation.
        - 'closed_formula' (numpy.ndarray): An array of the closed form formula.
    
    Returns:
    ________
    dict: A dictionary where the keys are tuples of (alpha, num_hidden_layer, cells_layer) and the values are dictionaries with various evaluation metrics for each model.
    """
    models = {}


    # We get closed form formula from base scenario. Notice that scenario 0 represents base scenario.
    closed_formula_base = test_scenarios_and_closed_formula[0]['closed_formula']

    # Iterate over all combinations of hyperparameters
    for a, c, n in product(alphas, cells_layer, num_hidden_layers):

        # Construct the name of the model based on the hyperparameters
        model_name = 'alpha' + '_' + str(a) + '_cells_' + str(c) + '_hidden_' + str(n) 

        # Print the name of the current model
        print(model_name)

        # Construct the path to the directory where the current model is stored
        path = PATH_MODELS +  model_name + '/'

        # Load the current model
        model = Deep_learning_models.Diff_learning_scaler.open(path)

        # We calculate the estimate for the base scenario
        pred_base = model.predict(test_scenarios_and_closed_formula[0]['scenario'], batch_size = 1)['y']

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the training data
        y_pred_train = model.predict(sim_scenarios_train, batch_size = len(sim_scenarios_train))['y']
        mse_train = mean_squared_error(y_pred_train, payoff_train)

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the cross-validation data
        y_pred_cv = model.predict(sim_scenarios_cv, batch_size = len(sim_scenarios_cv))['y']
        mse_cv = mean_squared_error(y_pred_cv, payoff_cv)

        

        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the training data
        spearman_train = spearmanr(y_pred_train - pred_base, closed_formula_train - closed_formula_base)
        ks_train = ks_2samp(y_pred_train - pred_base, closed_formula_train - closed_formula_base)

        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the cross-validation data
        spearman_cv = spearmanr(y_pred_cv - pred_base, closed_formula_train_cv - closed_formula_base)
        ks_cv = ks_2samp(y_pred_cv - pred_base, closed_formula_train_cv - closed_formula_base)

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
        
        for i in range(1,len(test_scenarios_and_closed_formula)):
            # Predict model for historical scenarios
            y_pred_test = model.predict(test_scenarios_and_closed_formula[i]['scenario'], batch_size = len(test_scenarios_and_closed_formula[i]['scenario']))['y']


            # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for historical scenarios
            spearman_test = spearmanr(y_pred_test - pred_base, test_scenarios_and_closed_formula[i]['closed_formula'] - closed_formula_base)
            ks_hist_test = ks_2samp(y_pred_test - pred_base, test_scenarios_and_closed_formula[i]['closed_formula'] - closed_formula_base)

            results_dict['spearman_' + test_scenarios_and_closed_formula[i]['scenario_name']] = spearman_test
            results_dict['ks_' + test_scenarios_and_closed_formula[i]['scenario_name']] = ks_hist_test

                    
        # Clear the output to keep the notebook clean
        clear_output()

    return models

def plot_model_results(models, alphas, cells_layer, num_hidden_layers, test_scenario_names, bayes_error_cv,
                       file_name, chart_name, chart_sub_name_FRTB, PATH_FIGS):

    plot_results = {}

    for c, n in product(cells_layer, num_hidden_layers):

        plot_results[(c,n)] = {}

        plot_results[(c,n)]['name'] = '{0} hidden layers, {1} cells'.format(n,c)


        plot_results[(c,n)]['mse_train'] = np.zeros(len(alphas))
        plot_results[(c,n)]['mse_cv'] = np.zeros(len(alphas))
        plot_results[(c,n)]['spearman_train'] = np.zeros(len(alphas))
        plot_results[(c,n)]['spearman_cv'] = np.zeros(len(alphas))
        plot_results[(c,n)]['ks_train'] = np.zeros(len(alphas))
        plot_results[(c,n)]['ks_cv'] = np.zeros(len(alphas))

        for test in test_scenario_names:
            plot_results[(c,n)]['spearman_' + test] = np.zeros(len(alphas))
            plot_results[(c,n)]['ks_' + test] = np.zeros(len(alphas))



    for m in models:

        a = models[m]['model_params']['alpha'] 
        n = models[m]['model_params']['hidden']
        c = models[m]['model_params']['cells']

        plot_results[(c,n)]['mse_train'][alphas.index(a)] = models[m]['mse_train']
        plot_results[(c,n)]['mse_cv'][alphas.index(a)] = models[m]['mse_cv']

        plot_results[(c,n)]['spearman_train'][alphas.index(a)] = models[m]['spearman_train']
        plot_results[(c,n)]['spearman_cv'][alphas.index(a)] = models[m]['spearman_cv']

        plot_results[(c,n)]['ks_train'][alphas.index(a)] = models[m]['ks_train']
        plot_results[(c,n)]['ks_cv'][alphas.index(a)] = models[m]['ks_cv']

        for test in test_scenario_names:           
                    
            plot_results[(c,n)]['spearman_' + test][alphas.index(a)] =  models[m]['spearman_' + test]
            plot_results[(c,n)]['ks_' + test][alphas.index(a)] = models[m]['ks_' + test]

    for p in plot_results:

        plt.plot(plot_results[p]['mse_cv'], '.-', label = plot_results[p]['name'])
        plt.axhline(y = bayes_error_cv, color = 'grey', linestyle = ':')
        
        
        plt.xticks(ticks = range(len(alphas)),labels = alphas)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'Diff. machine learning $\alpha$')
        plt.ylabel('Mean squared error in CV')

        plt.gcf().set_size_inches(6,4)  

        plt.title(chart_name)

        plt.savefig(PATH_FIGS + file_name + 'MSE_CV.pdf',bbox_inches ='tight')

    plt.figure()


    for p in plot_results:

        plt.plot(plot_results[p]['spearman_cv'], '.-', label = plot_results[p]['name'])
        
        
        plt.xticks(ticks = range(len(alphas)),labels = alphas)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axhline(y = 0.80, color = 'yellow', linestyle = ':')
        plt.axhline(y = 0.70, color = 'red', linestyle = ':')
        plt.xlabel(r'Diff. machine learning $\alpha$');   
        plt.ylabel(r'Spearman $\rho$ CV');   
        plt.ylim(0.6,1.0)

        plt.title(chart_name + chart_sub_name_FRTB)

        plt.savefig(PATH_FIGS + file_name + 'Spearman_CV.pdf',bbox_inches ='tight')

    plt.figure()


    for p in plot_results:

        plt.plot(plot_results[p]['ks_cv'], '.-', label = plot_results[p]['name'])
        
        plt.axhline(y = 0.12, color = 'red', linestyle = ':')
        plt.axhline(y = 0.09, color = 'yellow', linestyle = ':')
        plt.ylim(0.0,0.2)

        
        plt.xticks(ticks = range(len(alphas)),labels = alphas)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'Diff. machine learning $\alpha$');
        plt.ylabel(r'Kolmogorov-Smirnov statistic CV');   

        plt.title(chart_name + chart_sub_name_FRTB)

        plt.savefig(PATH_FIGS + file_name + 'KS_CV.pdf',bbox_inches ='tight')

    plt.figure()


    for test in test_scenario_names:


        for p in plot_results:

            plt.plot(plot_results[p]['spearman_' + test], '.-', label = plot_results[p]['name'])
            
            
            plt.xticks(ticks = range(len(alphas)),labels = alphas)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.axhline(y = 0.80, color = 'yellow', linestyle = ':')
            plt.axhline(y = 0.70, color = 'red', linestyle = ':')
            plt.xlabel(r'Diff. machine learning $\alpha$');   
            plt.ylabel(r'Spearman $\rho$ ' + test);   
            plt.ylim(0.6,1.0)

            plt.title(chart_name + test)

            plt.savefig(PATH_FIGS + file_name + 'Spearman_' + test + '.pdf',bbox_inches ='tight')

        plt.figure()

        for p in plot_results:

            plt.plot(plot_results[p]['ks_' + test], '.-', label = plot_results[p]['name'])
            
            plt.axhline(y = 0.12, color = 'red', linestyle = ':')
            plt.axhline(y = 0.09, color = 'yellow', linestyle = ':')

            
            plt.xticks(ticks = range(len(alphas)),labels = alphas)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel(r'Diff. machine learning $\alpha$');
            plt.ylabel(r'Kolmogorov-Smirnov statistic ' + test);   
            plt.ylim(0.0,0.2)

            plt.title(chart_name + test)

            plt.savefig(PATH_FIGS + file_name + 'KS_' + test + '.pdf',bbox_inches ='tight')

        plt.figure();
      





    







