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
import Enums
import pandas as pd
import Scenario_Simulation
from dateutil.relativedelta import relativedelta
import Option_formulas
import pickle
import gc

def TrainSetOfModels(PATH_MODELS, alphas, cells_layer, num_hidden_layers, hidden_activ_func, sim_scenarios_train, payoff_train, sens_train, epochs ,batch_size, valid_data = None):

    '''
    Trains a set of deep learning models.
    Inputs:
    _______
    * PATH_MODELS (str): Path where to save the set of models.
    * alphas (list(float)): Contains set of alpha (diff machine learning parameter).
    * cells_layer (list(int)): Contains set of number of cells per layer.
    * num_hidden_layers (list(int)): Contains set of number of hidden layers.
    * hidden_activ_func: Activation function of hidden layer
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
    Miscellanea.check_and_manage_path(PATH_MODELS)


    # Iterate over different values of hyperparameters
    for a, c, n in product(alphas, cells_layer, num_hidden_layers):

        # Set a random seed to ensure reproducibility of results
        tf.keras.utils.set_random_seed(9876)

         # Create a new directory to save the trained model and print a message indicating the current hyperparameter values
        print('FITTING MODEL: alpha: {0}, cells: {1}, num_hidden: {2}'.format(a, c, n))

        path = PATH_MODELS + 'alpha' + '_' + str(a) + '_cells_' + str(c) + '_hidden_' + str(n) + '/'
        os.mkdir(path)

         # Build a deep learning model with the specified number of hidden layers and cells per layer using the "build_dense_model" function
        dense_model = Deep_learning_models.build_dense_model(sim_scenarios_train.shape[1],n,c,hidden_activ_func,'linear')

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
                       test_scenarios_data, base_scenario_adj_option):
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
    closed_formula_plus_adj_train (np.ndarray): array of closed formula prices plus adjustments for training: 
        could be closed form formula for each scenario minus the closed form formula of the base scenario.
        could include the increment in the hedged basket. 
    closed_formula_plus_adj_cv (np.ndarray): array of closed formula prices plus adjustments for cross-validation:
        same as the previous comment for cv.
    model_adj_train (np.ndarray): array of model adjustments for training:
        Represents the adjustment to be made for the model prediction. Could be -closed_form_base and include 
        + basket_PL_inc. Since variance reduction is done with model predictions, it is not included here.
    model_adj_cv (np.ndarray): array of model adjustments for cross-validation.
    test_scenarios_data (list): list of dictionaries containing test scenarios with keys for scenario name, scenario data, 
                                closed formula prices plus adjustments, and model adjustments. Keys:
        - 'scenario_name' (str)
        - 'scenario' (numpy.ndarray): An array of simulated scenarios used for test.
        - 'closed_formula_plus_adj' (numpy.ndarray): An array of the closed form formula plus adjustments. See comments above.
        - 'model_adj': array of model adjustments not dependent on base scenario. See comments above.
        - 'base_scenario_closed_form_sens': only yo be included for base scenario.
    base_scenario_adj_option (str): flag indicating whether to adjust model with base scenario. Options: ('No','NPV','NPV_plus_sens')

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

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the training data
        y_pred_train = model.predict(sim_scenarios_train, batch_size = len(sim_scenarios_train))['y']
        mse_train = mean_squared_error(y_pred_train, payoff_train)

        # Compute the mean squared error between the predicted payoffs and the actual payoffs for the cross-validation data
        y_pred_cv = model.predict(sim_scenarios_cv, batch_size = len(sim_scenarios_cv))['y']
        

        mse_cv = mean_squared_error(y_pred_cv, payoff_cv)

        
        if (base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV) or (base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV_PLUS_SENS):
            # We calculate the estimate for the base scenario
            model_adj_base = - model.predict(test_scenarios_data[0]['scenario'], batch_size = 1)['y']
        else:
            model_adj_base = 0.0

        if (base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV_PLUS_SENS):
            # If adjustment option includes sensibilities
            # Compute model sensitivities for base scenario
            model_sens_base_scenario = model.predict(test_scenarios_data[0]['scenario'], batch_size = 1)['sens']
            model_adj_base_sens_cv = np.matmul(sim_scenarios_cv-test_scenarios_data[0]['scenario'],(test_scenarios_data[0]['base_scenario_closed_form_sens'] - model_sens_base_scenario).T).flatten()
            model_adj_base_sens_train = np.matmul(sim_scenarios_train-test_scenarios_data[0]['scenario'],(test_scenarios_data[0]['base_scenario_closed_form_sens'] - model_sens_base_scenario).T).flatten()
            
        else:
            model_adj_base_sens_cv = 0.0
            model_adj_base_sens_train = 0.0


        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the training data
        spearman_train, _ = spearmanr(y_pred_train + model_adj_base + model_adj_base_sens_train + model_adj_train, closed_formula_plus_adj_train)
        ks_train, _ = ks_2samp(y_pred_train + model_adj_base + model_adj_base_sens_train + model_adj_train, closed_formula_plus_adj_train)
        

        # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for the cross-validation data
        spearman_cv, _ = spearmanr(y_pred_cv + model_adj_base + model_adj_base_sens_cv + model_adj_cv, closed_formula_plus_adj_cv)
        ks_cv, _ = ks_2samp(y_pred_cv + model_adj_base + model_adj_base_sens_cv + model_adj_cv, closed_formula_plus_adj_cv)

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
        results_dict['cv_scenarios_closed_form'] = closed_formula_plus_adj_cv
        results_dict['cv_scenarios_model_results_plus_adjustments'] = y_pred_cv + model_adj_base + model_adj_base_sens_cv + model_adj_cv
        
        
        # Add the results for the current model to the dictionary of all models
        metrics[(a,n,c)] = results_dict
        
        for i in range(1,len(test_scenarios_data)):
            # Predict model for historical scenarios
            y_pred_test = model.predict(test_scenarios_data[i]['scenario'], batch_size = len(test_scenarios_data[i]['scenario']))['y']

            if (base_scenario_adj_option == Enums.Base_Scenario_Adj_Option.NPV_PLUS_SENS):
                # If adjustment option includes sensibilities
                # Compute model sensitivities for base scenario
                model_adj_base_sens_test = np.matmul(test_scenarios_data[i]['scenario']-test_scenarios_data[0]['scenario'],(test_scenarios_data[0]['base_scenario_closed_form_sens'] - model_sens_base_scenario).T).flatten()
            else:
                model_adj_base_sens_test = 0.0


            # Compute the Spearman correlation coefficient and the KS statistic between the predicted payoffs and the closed-formula prices for historical scenarios
            spearman_test, _ = spearmanr(y_pred_test + model_adj_base + model_adj_base_sens_test + test_scenarios_data[i]['model_adj'], test_scenarios_data[i]['closed_formula_plus_adj'])
            ks_hist_test, _ = ks_2samp(y_pred_test + model_adj_base + model_adj_base_sens_test + test_scenarios_data[i]['model_adj'], test_scenarios_data[i]['closed_formula_plus_adj'])

            results_dict['spearman_' + test_scenarios_data[i]['scenario_name']] = spearman_test
            results_dict['ks_' + test_scenarios_data[i]['scenario_name']] = ks_hist_test

            results_dict['cv_scenarios_closed_form_' + test_scenarios_data[i]['scenario_name']] =  test_scenarios_data[i]['closed_formula_plus_adj']

            results_dict['cv_scenarios_model_results_plus_adjustments_' + test_scenarios_data[i]['scenario_name']] = y_pred_test + model_adj_base + model_adj_base_sens_test + test_scenarios_data[i]['model_adj']
            

        # Clear the output to keep the notebook clean
        clear_output()
    
    mse_list = []
    mse_keys = []

    mse_list_zero_alpha = []
    mse_keys_zero_alpha = []


    for key in metrics:

        mse_keys += [key]
        mse_list += [metrics[key]['mse_cv']]   

        if key[0] == 0:

            mse_keys_zero_alpha += [key]
            mse_list_zero_alpha += [metrics[key]['mse_cv']]   



    # print('Best model:')
    # print(metrics[mse_keys[np.argmin(mse_list)]])

    return metrics, metrics[mse_keys[np.argmin(mse_list)]], metrics[mse_keys_zero_alpha[np.argmin(mse_list_zero_alpha)]]


def plot_model_results(metrics, best_model_metrics, best_model_zero_alpha_metrics,alphas, cells_layer, num_hidden_layers, test_scenario_names, bayes_error_cv,
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



    for m in metrics:

        a = metrics[m]['model_params']['alpha'] 
        n = metrics[m]['model_params']['hidden']
        c = metrics[m]['model_params']['cells']

        plot_results[(c,n)]['mse_train'][alphas.index(a)] = metrics[m]['mse_train']
        plot_results[(c,n)]['mse_cv'][alphas.index(a)] = metrics[m]['mse_cv']

        plot_results[(c,n)]['spearman_train'][alphas.index(a)] = metrics[m]['spearman_train']
        plot_results[(c,n)]['spearman_cv'][alphas.index(a)] = metrics[m]['spearman_cv']

        plot_results[(c,n)]['ks_train'][alphas.index(a)] = metrics[m]['ks_train']
        plot_results[(c,n)]['ks_cv'][alphas.index(a)] = metrics[m]['ks_cv']

        for test in test_scenario_names:           
                    
            plot_results[(c,n)]['spearman_' + test][alphas.index(a)] =  metrics[m]['spearman_' + test]
            plot_results[(c,n)]['ks_' + test][alphas.index(a)] = metrics[m]['ks_' + test]

    for p in plot_results:

        if (p[0] == best_model_metrics['model_params']['cells']) and (p[1] == best_model_metrics['model_params']['hidden']):  
            plt.plot(plot_results[p]['mse_cv'], '.-', label = plot_results[p]['name'], linewidth=3)
            
            index_best = alphas.index(best_model_metrics['model_params']['alpha'])

            plt.plot(index_best,plot_results[p]['mse_cv'][index_best], 'o',  markersize = 8, markerfacecolor='none', markeredgecolor='black')
            
        else:
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

        if (p[0] == best_model_metrics['model_params']['cells']) and (p[1] == best_model_metrics['model_params']['hidden']):  

            plt.plot(plot_results[p]['spearman_cv'], '.-', label = plot_results[p]['name'], linewidth=3)

            index_best = alphas.index(best_model_metrics['model_params']['alpha'])

            plt.plot(index_best,plot_results[p]['spearman_cv'][index_best], 'o',  markersize = 8, markerfacecolor='none', markeredgecolor='black')
        
        else:
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

        if (p[0] == best_model_metrics['model_params']['cells']) and (p[1] == best_model_metrics['model_params']['hidden']):  

            plt.plot(plot_results[p]['ks_cv'], '.-', label = plot_results[p]['name'], linewidth=3)
            index_best = alphas.index(best_model_metrics['model_params']['alpha'])

            plt.plot(index_best,plot_results[p]['ks_cv'][index_best], 'o',  markersize = 8, markerfacecolor='none', markeredgecolor='black')
        
        else:
            plt.plot(plot_results[p]['ks_cv'], '.-', label = plot_results[p]['name'])

        
        plt.axhline(y = 0.12, color = 'red', linestyle = ':')
        plt.axhline(y = 0.09, color = 'yellow', linestyle = ':')
        plt.ylim(0.0,0.3)

        
        plt.xticks(ticks = range(len(alphas)),labels = alphas)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'Diff. machine learning $\alpha$');
        plt.ylabel(r'Kolmogorov-Smirnov statistic CV');   

        plt.title(chart_name + chart_sub_name_FRTB)

        plt.savefig(PATH_FIGS + file_name + 'KS_CV.pdf',bbox_inches ='tight')

    plt.figure()


    for test in test_scenario_names:


        for p in plot_results:

            if (p[0] == best_model_metrics['model_params']['cells']) and (p[1] == best_model_metrics['model_params']['hidden']):

                plt.plot(plot_results[p]['spearman_' + test], '.-', label = plot_results[p]['name'], linewidth=3)

                index_best = alphas.index(best_model_metrics['model_params']['alpha'])

                plt.plot(index_best,plot_results[p]['spearman_' + test][index_best], 'o',  markersize = 8, markerfacecolor='none', markeredgecolor='black')
            
            else:

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

            if (p[0] == best_model_metrics['model_params']['cells']) and (p[1] == best_model_metrics['model_params']['hidden']):  

                plt.plot(plot_results[p]['ks_' + test], '.-', label = plot_results[p]['name'], linewidth=3)
                index_best = alphas.index(best_model_metrics['model_params']['alpha'])

                plt.plot(index_best,plot_results[p]['ks_' + test][index_best], 'o',  markersize = 8, markerfacecolor='none', markeredgecolor='black')
            else:
                plt.plot(plot_results[p]['ks_' + test], '.-', label = plot_results[p]['name'])
            
            plt.axhline(y = 0.12, color = 'red', linestyle = ':')
            plt.axhline(y = 0.09, color = 'yellow', linestyle = ':')

            
            plt.xticks(ticks = range(len(alphas)),labels = alphas)
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel(r'Diff. machine learning $\alpha$');
            plt.ylabel(r'Kolmogorov-Smirnov statistic ' + test);   
            plt.ylim(0.0,0.3)

            plt.title(chart_name + test)

            plt.savefig(PATH_FIGS + file_name + 'KS_' + test + '.pdf',bbox_inches ='tight')

        plt.figure()

    f = Miscellanea.plot_plat_charts(best_model_metrics['cv_scenarios_closed_form'], best_model_metrics['cv_scenarios_model_results_plus_adjustments'], 
                                     fig_tittle= 'Best model PLAT in cv')

    f.savefig(PATH_FIGS + file_name + 'Best_difflearn_model_PLAT_cv.pdf',bbox_inches ='tight')

    f = Miscellanea.plot_plat_charts(best_model_zero_alpha_metrics['cv_scenarios_closed_form'], best_model_zero_alpha_metrics['cv_scenarios_model_results_plus_adjustments'], 
                                     fig_tittle= r'Best model ($\lambda = 0$) PLAT in cv')

    f.savefig(PATH_FIGS + file_name + 'Best_traditional_model_PLAT_cv.pdf',bbox_inches ='tight')

    for test in test_scenario_names:

        f = Miscellanea.plot_plat_charts(best_model_metrics['cv_scenarios_closed_form_' + test], best_model_metrics['cv_scenarios_model_results_plus_adjustments_' + test], 
                                     fig_tittle= 'Best model PLAT ' + test + ' schocks')

        f.savefig(PATH_FIGS + file_name + 'Best_difflearn_model_PLAT_' + test + '.pdf',bbox_inches ='tight')

        f = Miscellanea.plot_plat_charts(best_model_zero_alpha_metrics['cv_scenarios_closed_form_' + test], best_model_zero_alpha_metrics['cv_scenarios_model_results_plus_adjustments_' + test], 
                                        fig_tittle= r'Best model ($\lambda = 0$) PLAT ' + test + ' schocks')

        f.savefig(PATH_FIGS + file_name + 'Best_traditional_model_PLAT_' + test + '.pdf',bbox_inches ='tight')


def PLAT_Analysis(base_scenario_dict, test_scenario_dict, train_scenario_dict, model_dict, plat_dict):
    """
    Performs PLAT analysis using deep learning models.

    This function builds a deep learning model based on specified parameters, prepares scenario data for
    base and test cases, and then fits the model to the training data. It uses various callbacks to adjust
    the scenarios based on different analysis options, and returns a dictionary of these callbacks.

    Parameters:
    base_scenario_dict (dict): Dictionary containing the base scenario data.
        Keys:
        - 'scenario_levels': Scenario levels for the base case.
        - 'closed_form_sens': Sensitivity analysis data for the base scenario.
        - 'closed_form_value': Closed form value for the base scenario (used in test scenario adjustments).
        - 'hedge_NPV': (if applicable) Net present value for the base scenario hedge.

    test_scenario_dict (dict): Dictionary containing the test scenario data.
        Keys:
        - 'scenario_name': Name of the test scenario.
        - 'scenario_levels': Scenario levels for the test case.
        - 'closed_form_value': Closed form value for the test scenario.
        - 'hedge_NPV': (if applicable) Net present value for the test scenario hedge.

    train_scenario_dict (dict): Dictionary containing the training scenario data.
        Keys:
        - 'scenario_levels': Scenario levels for training.
        - 'payoff': Payoff data for training.
        - 'pathwise_derivs': Pathwise derivatives for training.

    model_dict (dict): Dictionary containing model configuration parameters.
        Keys:
        - 'input_dim': Input dimension for the model.
        - 'num_hidden_layers': Number of hidden layers in the model.
        - 'num_cells_hidden_layers': Number of cells in each hidden layer.
        - 'hidden_layer_activation': Activation function for hidden layers.
        - 'output_layer_activation': Activation function for the output layer.
        - 'alpha': Learning rate.
        - 'batch_size': Batch size for training.
        - 'num_epochs': Number of epochs for training.
        - 'verbose': Verbosity mode.

    plat_dict (dict): Dictionary containing PLAT analysis options and parameters.
        Keys:
        - 'plat_analysis_option': Selected option for PLAT analysis (e.g., NAIVE).
        - 'num_batches_callback': Number of batches for the callback.

    Returns:
    dict: A dictionary of PLAT callbacks for different analysis scenarios, including those for 
    'NO', 'NPV', 'SENS', 'Hedge_NO', 'Hedge_NPV', and 'Hedge_SENS' adjustments.
    """

    

    model = Deep_learning_models.build_diff_learning_model(model_dict['input_dim'],
            model_dict['num_hidden_layers'],
            model_dict['num_cells_hidden_layers'],
            model_dict['hidden_layer_activation'],
            model_dict['output_layer_activation'],
            model_dict['alpha'])


        
    if plat_dict['plat_analysis_option'] == Enums.Plat_Analysis_Option.CONVEXITY:
        linear_term_train = np.matmul((train_scenario_dict['scenario_levels'] - base_scenario_dict['scenario_levels']),
                                      base_scenario_dict['closed_form_sens']).flatten()
        #linear_term_test = np.matmul((test_scenario_dict['scenario_levels'] - base_scenario_dict['scenario_levels']),
        #                              base_scenario_dict['closed_form_sens']).flatten()
        sens_adj = base_scenario_dict['closed_form_sens']
        
    else: 
        linear_term_train = 0.0
        linear_term_test = 0.0
        sens_adj = 0.0

                      
    
    plat_callback =  Deep_learning_models.Plat_Callback(base_scenario_dict, test_scenario_dict, model,
        plat_dict['plat_analysis_option'], plat_dict['num_batches_callback'])

    
    

    
    model.fit(train_scenario_dict['scenario_levels'],
            train_scenario_dict['payoff'] - linear_term_train,
            train_scenario_dict['pathwise_derivs'] - sens_adj, model_dict['batch_size'],
            model_dict['num_epochs'], None,
            [plat_callback],
            model_dict['verbose'])


    return plat_callback



class plat_orquestrator:

    def __init__(self, training_data_generation_dict, machine_learning_models_dict,
                 risk_management_dict, hist_data):

        self.gaussian_model_dict = Miscellanea.deep_copy_dict_with_arrays(training_data_generation_dict['gaussian_model_dict'])
        self.base_scenario_dict = Miscellanea.deep_copy_dict_with_arrays(training_data_generation_dict['base_scenario_dict'])
        self.simulation_dict = Miscellanea.deep_copy_dict_with_arrays(training_data_generation_dict['simulation_dict'])
        self.contract_data_dict = Miscellanea.deep_copy_dict_with_arrays(training_data_generation_dict['contract_data_dict'])
        self.machine_learning_models_dict = Miscellanea.deep_copy_dict_with_arrays(machine_learning_models_dict)

        self.simulation_dict['number_of_scenarios'] = machine_learning_models_dict['num_training_examples_payoff_reset']

        self.hist_data = hist_data.drop_duplicates(keep='first')

        self.hist_schocks_10d = pd.DataFrame(data = np.log(self.hist_data.iloc[10:].values/ self.hist_data.iloc[0:-10].values),
                            index =  self.hist_data.index[10:], columns = self.hist_data.columns)


        self.hist_schocks_1d = pd.DataFrame(data = np.log(self.hist_data.iloc[1:].values/ self.hist_data.iloc[0:-1].values),
                            index =  self.hist_data.index[1:], columns = self.hist_data.columns)

        self.risk_management_dict = Miscellanea.deep_copy_dict_with_arrays(risk_management_dict)

        self.init_and_maturity_dates = self.generate_dates(self.risk_management_dict['initial_date'],
                                self.risk_management_dict['end_date'],
                                self.risk_management_dict['maturity_in_months_at_reset_dates'])

        self.risk_management_dict['initial_date'] = self.init_and_maturity_dates[0]

        self.model = None

    def generate_dates(self, start_date, end_date, month_interval):
        """
        Generates dates every 'month_interval' months between 'start_date' and 'end_date'.
        :param start_date: The starting date.
        :param end_date: The ending date.
        :param month_interval: The interval in months.
        :return: A list of dates.
        """
        dates = []
        current_date = end_date
        while current_date > start_date:
            dates.append(current_date)
            current_date -= relativedelta(months=month_interval)

        dates.append(start_date)

        dates = self.hist_data.index[self.hist_data.index.get_indexer(dates, method='nearest')]

        return np.array(dates[::-1])  # Reverse the list to have it in ascending order


    def compute_historical_correlation(self, date):

        date_index = self.hist_schocks_1d.index.get_indexer([date])[0]

        return self.hist_schocks_1d.iloc[date_index- \
            self.base_scenario_dict['num_escen_correl_matrix']+1:date_index+1,
            self.base_scenario_dict['spot_indexes']].corr().values

    def update_base_scenario(self, date):

        date_index = self.hist_data.index.get_indexer([date])[0]

        self.base_scenario_dict['spots'] = \
            self.hist_data.iloc[date_index,self.base_scenario_dict['spot_indexes']].values

        self.base_scenario_dict['vols'] = \
            self.hist_data.iloc[date_index,self.base_scenario_dict['vol_indexes']].values / 100.0

        self.base_scenario_dict['base_scenario'] =  np.zeros((1,self.gaussian_model_dict['n_components']))
        self.base_scenario_dict['base_scenario'][0,self.base_scenario_dict['spot_indexes']] = self.base_scenario_dict['spots'] 
        self.base_scenario_dict['base_scenario'][0,self.base_scenario_dict['vol_indexes']] = self.base_scenario_dict['vols'] 
         

        self.base_scenario_dict['correlations'] = \
            self.compute_historical_correlation(date)

    def compute_indiv_strikes(self, date):

        last_fixing_date =  self.init_and_maturity_dates[self.init_and_maturity_dates <= date][-1]

        date_index = self.hist_data.index.get_indexer([last_fixing_date])[0]

        return self.hist_data.iloc[date_index,self.base_scenario_dict['spot_indexes']].values


    def update_payoff(self, date):
        # update_base_scenario to be called before

        next_maturity_date = self.init_and_maturity_dates[self.init_and_maturity_dates > date][0]

        indiv_strikes =  self.compute_indiv_strikes(date)

        self.contract_data_dict['indiv_strikes'] = indiv_strikes

        maturity_years = ((next_maturity_date - date).days)/365.25

        self.contract_data_dict['payoff'] = \
            self.contract_data_dict["generic_payoff"]( \
            indiv_strikes_value = indiv_strikes)


        self.contract_data_dict['closed_form_formula'] = lambda MktData: self.contract_data_dict['generic_closed_form_formula']( \
                    indiv_strikes_value = indiv_strikes, maturity = maturity_years)(
                        spot_t = MktData[:,self.base_scenario_dict['spot_indexes']],
                        vol_t =  MktData[:,self.base_scenario_dict['vol_indexes']],
                        rfr = self.base_scenario_dict['rfr'],
                        divs = self.base_scenario_dict['divs'],
                        correl = self.base_scenario_dict['correlations'])


        self.contract_data_dict['ttm'] = maturity_years

    def update_hedge(self, date):

        # update_base_scenario and update_payoff to be called before

        futs = []
        calls = []

        if self.risk_management_dict['hedge_maturity_in_years'] is not None:
            hedge_maturity = self.risk_management_dict['hedge_maturity_in_years']
        else:
            hedge_maturity = self.contract_data_dict['ttm']

        for i in range(int(self.gaussian_model_dict['n_components']/2)):
            loop_i = i

            if self.risk_management_dict['fut_hedge'][i]:
                futs += [lambda MktData, i=loop_i: Option_formulas.FutureTF(MktData[:,self.base_scenario_dict['spot_indexes'][i]],
                                    self.contract_data_dict['indiv_strikes'][i],
                                    hedge_maturity, self.base_scenario_dict['rfr'], 
                                    self.base_scenario_dict['divs'])]
                
            if self.risk_management_dict['call_hedge'][i]:
                calls += [lambda MktData, i=loop_i: Option_formulas.BlackScholesTF(MktData[:,self.base_scenario_dict['spot_indexes'][i]],
                                    self.contract_data_dict['indiv_strikes'][i], hedge_maturity,
                                    self.base_scenario_dict['rfr'], self.base_scenario_dict['divs'],
                                    MktData[:,self.base_scenario_dict['vol_indexes'][i]], True)]

        hedge_instruments = futs + calls   

        self.hedge = Option_formulas.Basket(hedge_instruments, self.contract_data_dict['closed_form_formula'] )

        self.hedge.compute_hedge(self.base_scenario_dict['base_scenario'])


    def compute_training_data(self, date):

        self.update_base_scenario(date)
        self.update_payoff(date)


        date_index = self.hist_schocks_1d.index.get_indexer([date])[0]


        dict_1d_results = Scenario_Simulation.calibrate_hist_data_simulate_training_data(
            gaussian_model_dict= self.gaussian_model_dict,
            base_scenario_dict= self.base_scenario_dict,
            simulation_dict= self.simulation_dict,
            contract_data_dict= self.contract_data_dict,
            hist_schocks_data= self.hist_schocks_1d.iloc[date_index -
            self.gaussian_model_dict['n_examples']:date_index].values)

        date_index = self.hist_schocks_10d.index.get_indexer([date])[0]

        dict_10d_results = Scenario_Simulation.calibrate_hist_data_simulate_training_data(
            gaussian_model_dict= self.gaussian_model_dict,   
            base_scenario_dict= self.base_scenario_dict,
            simulation_dict= self.simulation_dict,
            contract_data_dict= self.contract_data_dict,
            hist_schocks_data= self.hist_schocks_10d.iloc[date_index -
            self.gaussian_model_dict['n_examples']:date_index].values)

        train_data_keys = ['sim_scenario_levels', 'payoff', 'pathwise_derivs']

        dict_1d_results_train = {k:dict_1d_results[k] for k in train_data_keys}
        dict_10d_results_train = {k:dict_10d_results[k] for k in train_data_keys}


        dict_mixed_data_train = Miscellanea.shuffle_arrays_in_dict(
            Miscellanea.concat_dict_containing_np_arrays([dict_1d_results_train,dict_10d_results_train]))


        return dict_mixed_data_train


    def train_model(self, date):

        is_reset_date = date in self.init_and_maturity_dates

        if not is_reset_date:
            num_examples = self.machine_learning_models_dict['num_training_examples_daily_calc']
            num_epochs = self.machine_learning_models_dict['num_epochs_daily_calc']
        else:
            num_examples = self.machine_learning_models_dict['num_training_examples_payoff_reset']
            num_epochs = self.machine_learning_models_dict['num_epochs_payoff_reset']

            del self.model
            gc.collect()

            self.model = Deep_learning_models.build_diff_learning_model( \
                input_shape = self.gaussian_model_dict['n_components'],
                num_hidden_layers = self.machine_learning_models_dict['nb_hidden_layers'],
                num_neurons_hidden_layers = self.machine_learning_models_dict['nd_cells_hidden_layer'],
                hidden_layer_activation = self.machine_learning_models_dict['hidden_layer_activation'],
                output_layer_activation = self.machine_learning_models_dict['output_layer_activation'],
                alpha = self.machine_learning_models_dict['alpha'])

        dict_mixed_data_train = self.compute_training_data(date)


        self.model.fit(dict_mixed_data_train['sim_scenario_levels'][0:num_examples],
            dict_mixed_data_train['payoff'][0:num_examples],
            dict_mixed_data_train['pathwise_derivs'][0:num_examples],
            self.machine_learning_models_dict['batch_size'],
            num_epochs, None, None,
            self.machine_learning_models_dict['verbose'])

    def train_models_for_all_dates(self):
        
        Miscellanea.check_and_manage_path(self.machine_learning_models_dict['MODELS_PATH'])

        index_begin_date = self.hist_data.index.get_indexer([self.init_and_maturity_dates[0]])[0]
        
        for i in range(index_begin_date,len(self.hist_data)-2):
            
            date = self.hist_data.index[i] 
            
            print("Fitting model for date " + date.strftime('%d-%m-%Y'))
            
            path = self.machine_learning_models_dict['MODELS_PATH'] + 'PLAT_MODELS/' + date.strftime('%Y-%m-%d/') 
            os.mkdir(path)
            
            self.train_model(date)
            self.model.save(path)
            
            clear_output()
    
    def continue_training_models(self):

        last_fitted_date = Miscellanea.get_latest_non_empty_subfolder_and_delete_empty(self.machine_learning_models_dict['MODELS_PATH'] + 'PLAT_MODELS/')

        index_begin_date = self.hist_data.index.get_indexer([pd.Timestamp(last_fitted_date)])[0] + 1
        
        self.model = Deep_learning_models.Diff_learning_scaler.open(self.machine_learning_models_dict['MODELS_PATH'] + 'PLAT_MODELS/' + last_fitted_date + '/')

        for i in range(index_begin_date,len(self.hist_data)-2):
            
            date = self.hist_data.index[i] 
            
            print("Fitting model for date " + date.strftime('%d-%m-%Y'))
            
            path = self.machine_learning_models_dict['MODELS_PATH'] +  'PLAT_MODELS/' +date.strftime('%Y-%m-%d/') 
            os.mkdir(path)
            
            self.train_model(date)
            self.model.save(path)
            
            clear_output()


            
    def run_plat_analysis(self):
        
        self.hpl = []
        self.hpl_with_hedge = []
        self.rtpl_naive = []
        self.rtpl_naive_with_hedge = []
        self.rtpl_npv = []
        self.rtpl_npv_with_hedge = []
        self.rtpl_sens = []
        self.rtpl_sens_with_hedge = []

        
        self.dates = []
        
        index_begin_date = self.hist_data.index.get_indexer([self.init_and_maturity_dates[0]])[0]


        for i in range(index_begin_date,len(self.hist_data)-2):
            
            date = self.hist_data.index[i] 
            
            print("Computing PLAT analysis for date " + date.strftime('%d-%m-%Y'))
            
            self.dates += [date]
            
            self.update_base_scenario(date)
            self.update_payoff(date)
            self.update_hedge(date)

            path = self.machine_learning_models_dict['MODELS_PATH'] + 'PLAT_MODELS/' + date.strftime('%Y-%m-%d/') 
            
            model = Deep_learning_models.Diff_learning_scaler.open(path)

            base_scenario = self.base_scenario_dict['base_scenario']        
            model_prediction_base_scenario = model.predict(base_scenario, batch_size = 1) 
            closed_form_formula_base_scenario = self.contract_data_dict['closed_form_formula'](base_scenario).numpy()[0]
            npv_hedge_base_scenario = self.hedge.value_basket(base_scenario)[0]
            
            next_date = self.hist_data.index[i+1]
            self.update_base_scenario(next_date)
            next_date_scenario = self.base_scenario_dict['base_scenario']
            model_prediction_next_date_scenario = model.predict(next_date_scenario, batch_size = 1) 
            closed_form_formula_next_date_scenario = self.contract_data_dict['closed_form_formula'](next_date_scenario).numpy()[0]
            npv_hedge_next_date_scenario = self.hedge.value_basket(next_date_scenario)[0]

            self.hpl += [closed_form_formula_next_date_scenario - closed_form_formula_base_scenario]
            self.hpl_with_hedge += [closed_form_formula_next_date_scenario - closed_form_formula_base_scenario + npv_hedge_next_date_scenario - npv_hedge_base_scenario]

     
            if self.simulation_dict['Simulate_Var_Red_Payoff'] == Enums.Simulate_Var_Red_Payoff.NO:

                self.rtpl_naive += [model_prediction_next_date_scenario['y'][0] - closed_form_formula_base_scenario]
                self.rtpl_naive_with_hedge += [model_prediction_next_date_scenario['y'][0] - closed_form_formula_base_scenario + 
                                              npv_hedge_next_date_scenario - npv_hedge_base_scenario]

                self.rtpl_npv += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0]]
                self.rtpl_npv_with_hedge += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0] + 
                                             npv_hedge_next_date_scenario - npv_hedge_base_scenario]

                self.rtpl_sens += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0] +
                    np.sum((next_date_scenario - base_scenario)*(self.hedge.grad.T - model_prediction_base_scenario['sens']))]

                self.rtpl_sens_with_hedge += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0] +
                    np.sum((next_date_scenario - base_scenario)*(self.hedge.grad.T - model_prediction_base_scenario['sens'])) +
                    npv_hedge_next_date_scenario - npv_hedge_base_scenario]
                
            elif self.simulation_dict['Simulate_Var_Red_Payoff'] == Enums.Simulate_Var_Red_Payoff.YES:

                self.rtpl_naive += [model_prediction_next_date_scenario['y'][0]]
                self.rtpl_naive_with_hedge += [model_prediction_next_date_scenario['y'][0] + 
                                              npv_hedge_next_date_scenario - npv_hedge_base_scenario]

                self.rtpl_npv += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0]]
                self.rtpl_npv_with_hedge += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0] + 
                                             npv_hedge_next_date_scenario - npv_hedge_base_scenario]

                self.rtpl_sens += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0] +
                    np.sum((next_date_scenario - base_scenario)*(self.hedge.grad.T - model_prediction_base_scenario['sens']))]

                self.rtpl_sens_with_hedge += [model_prediction_next_date_scenario['y'][0] - model_prediction_base_scenario['y'][0] +
                    np.sum((next_date_scenario - base_scenario)*(self.hedge.grad.T - model_prediction_base_scenario['sens'])) +
                    npv_hedge_next_date_scenario - npv_hedge_base_scenario]
                
            clear_output()
                
        self.hpl_with_hedge = np.array(self.hpl_with_hedge)
        self.rtpl_naive = np.array(self.rtpl_naive)
        self.rtpl_naive_with_hedge = np.array(self.rtpl_naive_with_hedge)
        self.rtpl_npv = np.array(self.rtpl_npv)
        self.rtpl_npv_with_hedge = np.array(self.rtpl_npv_with_hedge)
        self.rtpl_sens = np.array(self.rtpl_sens)
        self.rtpl_sens_with_hedge = np.array(self.rtpl_sens_with_hedge)

        self.save_analysis_results(self.machine_learning_models_dict['MODELS_PATH'] + self.machine_learning_models_dict['plat_analysis_subfolder'])

    def save_analysis_results(self, save_path):
            """
            Saves analysis results to the specified path.

            Parameters:
            save_path (str): The path to save the analysis results to.
            """
            # Ensure the save directory exists
            Miscellanea.check_and_manage_path(save_path)

            # List of attributes to save using pickle
            pickle_attributes = ['dates']
            # List of numpy array attributes to save
            numpy_attributes = ['hpl', 'hpl_with_hedge', 'rtpl_naive', 'rtpl_naive_with_hedge', 
                                'rtpl_npv', 'rtpl_npv_with_hedge', 'rtpl_sens', 'rtpl_sens_with_hedge']

            # Save attributes using pickle
            for attr in pickle_attributes:
                with open(os.path.join(save_path, f'{attr}.pkl'), 'wb') as file:
                    pickle.dump(getattr(self, attr, None), file)

            # Save numpy array attributes
            for attr in numpy_attributes:
                np.save(os.path.join(save_path, f'{attr}.npy'), getattr(self, attr, None))

            print(f"Analysis results saved to {save_path}")

    def load_analysis_results(self):
        """
        Loads analysis results from the specified path.

        """
        load_path = self.machine_learning_models_dict['MODELS_PATH'] + self.machine_learning_models_dict['plat_analysis_subfolder']

        # Check if the load directory exists
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Load path {load_path} does not exist.")

        # List of attributes to load using pickle
        pickle_attributes = ['dates']
        # List of numpy array attributes to load
        numpy_attributes = ['hpl', 'hpl_with_hedge', 'rtpl_naive', 'rtpl_naive_with_hedge', 
                            'rtpl_npv', 'rtpl_npv_with_hedge', 'rtpl_sens', 'rtpl_sens_with_hedge']

        # Load attributes using pickle
        for attr in pickle_attributes:
            with open(os.path.join(load_path, f'{attr}.pkl'), 'rb') as file:
                setattr(self, attr, pickle.load(file))

        # Load numpy array attributes
        for attr in numpy_attributes:
            setattr(self, attr, np.load(os.path.join(load_path, f'{attr}.npy')))

        print(f"Analysis results loaded from {load_path}")


                    
                
        
            



              











