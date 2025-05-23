import numpy as np
from numpy.linalg import cholesky
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle  
import Enums



class GaussianModel:
    """
    A model that wraps the Gaussian Mixture Model from scikit-learn.
    It allows for fitting the model to data and simulating new data based on the fit.
    """
    
    def __init__(self):
        """
        Initialize the GaussianModel instance.
        """
        self.model = None

    def fit(self, n_components, data, model_seed = None ):
        """
        Fit the Gaussian Mixture Model to the given data.
        
        Parameters:
        - n_components (int): The number of mixture components.
        - model_seed (int): The seed for the random initialization. Will be used both in fitting and simulation
        - data (array-like): The data to fit the model to.
        """
        
        self.model = GaussianMixture(n_components=n_components, random_state=model_seed)
        
            
        self.model.fit(data)

    def simulate(self, num_sims):
        """
        Simulate data based on the fitted Gaussian Mixture Model.
        
        Parameters:
        - num_sims (int): The number of data points to simulate.

        
        Returns:
        - array-like: The simulated data.
        """
        if self.model is None:
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method before simulating.")
        
        samples = self.model.sample(num_sims)[0]
        
        return shuffle(samples, random_state=self.model.random_state)
        


def simulate_mkt_risk_scenario_shifts(scenario_simulator, number_of_scenarios):
    """
    Simulates market risk scenarios based on the provided scenario simulator.

    Parameters:
    - scenario_simulator (object): An instance of a scenario simulator that must have a `simulate` method.
    - number_of_scenarios (int): The number of market risk scenarios to simulate.

    Returns:
    - array-like: Simulated market risk scenarios.
    """
    
    # Check if the given simulator object has a simulate method
    if not hasattr(scenario_simulator, "simulate"):
        raise ValueError("Provided scenario_simulator does not have a 'simulate' method.")
    
    # Use the simulate method of the provided scenario_simulator to generate the scenarios
    return scenario_simulator.simulate(number_of_scenarios)



def generate_mkt_risk_scenarios_levels(base_scenario, scenario_shifts, generation_option):
    """
    Simulate market risk scenarios based on a given option.

    Parameters:
    - base_scenario (np.array): 1D numpy array representing the initial or base values for each risk factor.
    - scenario_shifts (np.array): 2D numpy array where each row represents a market shock scenario 
                                  and columns represent shocks for each risk factor.
    - generation_option (MktRisk_Scenarios_Generation_Option): Indicates the type of simulation to be applied. 
      Options could be MktRisk_Scenarios_Generation_Option.EXPONENTIAL or MktRisk_Scenarios_Generation_Option.LINEAR.

    Returns:
    - np.array: A 2D array of simulated market scenarios based on the given option.
    
    Raises:
    - ValueError: If the provided option is neither 'EXPONENTIAL' nor 'LINEAR'.
                 If the number of risk factors in base_scenario doesn't match with the columns in scenario_shifts.
    - TypeError: If base_scenario is not a 1D array or scenario_shifts is not a 2D array.
    
    Example:
    >>> base = np.array([100, 150])
    >>> shifts = np.array([[0.01, -0.02], [-0.01, 0.02]])
    >>> generate_mkt_risk_scenarios(base, shifts, MktRisk_Scenarios_Generation_Option.EXPONENTIAL)
    [[101.00501671, 147.02937426], [99.00498339, 152.97062574]]
    """

    # Check for correct base_scenario type and dimension
    if not isinstance(base_scenario, np.ndarray) or base_scenario.ndim != 1:
        raise TypeError("base_scenario must be a 1D numpy array.")
    
    # Check for correct scenario_shifts type and dimension
    if not isinstance(scenario_shifts, np.ndarray) or scenario_shifts.ndim != 2:
        raise TypeError("scenario_shifts must be a 2D numpy array.")

    # Check if the number of risk factors match
    if base_scenario.shape[0] != scenario_shifts.shape[1]:
        raise ValueError("Number of risk factors in base_scenario doesn't match with the columns in scenario_shifts.")

    # Exponential simulation
    if generation_option == Enums.MktRisk_Scenarios_Generation_Option.EXPONENTIAL:
        return base_scenario * np.exp(scenario_shifts)
    
    # Linear simulation
    elif generation_option == Enums.MktRisk_Scenarios_Generation_Option.LINEAR:
        return base_scenario + scenario_shifts

    # Handle unexpected generation_option
    else:
        raise ValueError(f"Unsupported generation_option: {generation_option}. Supported options are: EXPONENTIAL, LINEAR.")

import numpy as np
from scipy.linalg import cholesky
import tensorflow as tf
from enum import Enum



def simulate_product_discounted_payoff(mkt_risk_scenario_levels, spot_indexes, vol_indexes, correlations, rfr, divs, ttm, random_seed, payoff, tf_option):
    """
    Simulate product payoff.

    Parameters explained in previous comments.

    Returns:
    - np.ndarray: Simulated product payoffs.
    """
    
    np.random.seed(random_seed)

    chol = cholesky(correlations)

    num_sims = mkt_risk_scenario_levels.shape[0]

    brow_ind = np.sqrt(ttm) * np.random.normal(loc=0.0, scale=1.0, size=(num_sims, len(spot_indexes)))

    brow_correl = np.matmul(brow_ind, chol.T)

    if tf_option == Enums.Include_Tensorflow_Calcs_option.NO:

        spots_t = mkt_risk_scenario_levels[:, spot_indexes] * np.exp((rfr - divs - 0.5 *mkt_risk_scenario_levels[:, vol_indexes]**2) * ttm + mkt_risk_scenario_levels[:, vol_indexes] * brow_correl)

        return payoff(spots_t)*np.exp(-rfr*ttm)

    elif tf_option == Enums.Include_Tensorflow_Calcs_option.YES:
    
        spots_t = mkt_risk_scenario_levels[:, spot_indexes] * tf.exp((rfr - divs - 0.5 *mkt_risk_scenario_levels[:, vol_indexes]**2) * ttm + mkt_risk_scenario_levels[:, vol_indexes] * brow_correl)

        return payoff(spots_t)*tf.exp(-rfr*ttm)
    else:
        raise ValueError("Invalid value for tf_option.")
    

def basket_option(spots_t, indiv_strikes, option_strike, tf_option):
    """
    Calculate the payoff for a Basket Option.

    Parameters explained in previous comments.

    Returns:
    - np.ndarray or tf.Tensor: Calculated option payoffs.
    """

    if tf_option == Enums.Include_Tensorflow_Calcs_option.YES:
        return tf.maximum(tf.exp(tf.reduce_mean(tf.math.log(spots_t/indiv_strikes), axis=1)) - option_strike, 0)
    elif tf_option == Enums.Include_Tensorflow_Calcs_option.NO:
        return np.maximum(np.exp(np.mean(np.log(spots_t/indiv_strikes), axis=1)) - option_strike, 0)
    else:
        raise ValueError("Invalid value for tf_option.")


def calibrate_hist_data_simulate_training_data(gaussian_model_dict, base_scenario_dict, simulation_dict, contract_data_dict, hist_schocks_data):
    """
    Calibrates a Gaussian model using historical shocks data and then simulates training data for market risk scenarios.

    Parameters:
    - gaussian_model_dict: Dictionary containing parameters for the Gaussian model.
    - base_scenario_dict: Dictionary defining the base scenario.
    - simulation_dict: Dictionary containing simulation settings.
    - contract_data_dict: Dictionary containing contract-related data.
    - hist_schocks_data: Historical shocks data.


    Returns:
    - Dictionary containing simulated scenario shifts, scenario levels, discounted payoff, and pathwise derivatives.
    """
    
    # Initialize and calibrate Gaussian model
    gm_model = GaussianModel()
    gm_model.fit(n_components=gaussian_model_dict['n_components'], data=hist_schocks_data, model_seed=gaussian_model_dict['gaussian_model_seed'])
    
    # Simulate market risk scenario shifts
    sim_scenario_shifts = simulate_mkt_risk_scenario_shifts(scenario_simulator=gm_model, number_of_scenarios=simulation_dict['number_of_scenarios'])
    
    # Define base scenario
    base_scenario = np.zeros(hist_schocks_data.shape[1])
    base_scenario[base_scenario_dict['spot_indexes']] = base_scenario_dict['spots']
    base_scenario[base_scenario_dict['vol_indexes']] = base_scenario_dict['vols']
    
    # Generate market risk scenario levels for historical shocks applied to base scenario
    hist_scenario_levels = generate_mkt_risk_scenarios_levels(base_scenario=base_scenario, 
                                                             scenario_shifts=hist_schocks_data, 
                                                             generation_option=simulation_dict['shocks_generation_option'])
    

    # Generate market risk scenario levels for simulated data
    sim_scenario_levels = generate_mkt_risk_scenarios_levels(base_scenario=base_scenario, 
                                                             scenario_shifts=sim_scenario_shifts, 
                                                             generation_option=simulation_dict['shocks_generation_option'])
    
    # Check for TensorFlow calculations inclusion option and perform calculations accordingly
    if simulation_dict['tf_generation_option'] == Enums.Include_Tensorflow_Calcs_option.YES:
        sim_scenario_levels_TF = tf.constant(sim_scenario_levels)

        with tf.GradientTape() as tape:
            tape.watch(sim_scenario_levels_TF)
            discounted_payoff = simulate_product_discounted_payoff(mkt_risk_scenario_levels=sim_scenario_levels_TF,
                                                                   spot_indexes=base_scenario_dict['spot_indexes'], 
                                                                   vol_indexes=base_scenario_dict['vol_indexes'],
                                                                   correlations=base_scenario_dict['correlations'], 
                                                                   rfr=base_scenario_dict['rfr'],
                                                                   divs=base_scenario_dict['divs'],
                                                                   ttm=contract_data_dict['ttm'],
                                                                   random_seed=simulation_dict['simulation_seed'],
                                                                   payoff=contract_data_dict['payoff'],
                                                                   tf_option=simulation_dict['tf_generation_option'])
            
            if simulation_dict['Simulate_Var_Red_Payoff'] == Enums.Simulate_Var_Red_Payoff.YES:

                base_scenario_repeated = np.repeat(base_scenario.reshape(1,-1), simulation_dict['number_of_scenarios'], axis = 0)

                discounted_payoff_base = simulate_product_discounted_payoff(mkt_risk_scenario_levels=base_scenario_repeated,
                                                                   spot_indexes=base_scenario_dict['spot_indexes'], 
                                                                   vol_indexes=base_scenario_dict['vol_indexes'],
                                                                   correlations=base_scenario_dict['correlations'], 
                                                                   rfr=base_scenario_dict['rfr'],
                                                                   divs=base_scenario_dict['divs'],
                                                                   ttm=contract_data_dict['ttm'],
                                                                   random_seed=simulation_dict['simulation_seed'],
                                                                   payoff=contract_data_dict['payoff'],
                                                                   tf_option=simulation_dict['tf_generation_option'])
                
                discounted_payoff = discounted_payoff - discounted_payoff_base

            
        # Compute pathwise derivatives
        pathwise_derivs = tape.gradient(discounted_payoff, sim_scenario_levels_TF).numpy()
        discounted_payoff = discounted_payoff.numpy() 

    elif simulation_dict['tf_generation_option'] == Enums.Include_Tensorflow_Calcs_option.NO:
        discounted_payoff = simulate_product_discounted_payoff(mkt_risk_scenario_levels=sim_scenario_levels,
                                                               spot_indexes=base_scenario_dict['spot_indexes'], 
                                                               vol_indexes=base_scenario_dict['vol_indexes'],
                                                               correlations=base_scenario_dict['correlations'], 
                                                               rfr=base_scenario_dict['rfr'],
                                                               divs=base_scenario_dict['divs'],
                                                               ttm=contract_data_dict['ttm'],
                                                               random_seed=simulation_dict['simulation_seed'],
                                                               payoff=contract_data_dict['payoff'],
                                                               tf_option=simulation_dict['tf_generation_option'])
        
        if simulation_dict['Simulate_Var_Red_Payoff'] == Enums.Simulate_Var_Red_Payoff.YES:

            base_scenario_repeated = np.repeat(base_scenario.reshape(1,-1), simulation_dict['number_of_scenarios'], axis = 0)

            discounted_payoff_base = simulate_product_discounted_payoff(mkt_risk_scenario_levels=base_scenario_repeated,
                                                                spot_indexes=base_scenario_dict['spot_indexes'], 
                                                                vol_indexes=base_scenario_dict['vol_indexes'],
                                                                correlations=base_scenario_dict['correlations'], 
                                                                rfr=base_scenario_dict['rfr'],
                                                                divs=base_scenario_dict['divs'],
                                                                ttm=contract_data_dict['ttm'],
                                                                random_seed=simulation_dict['simulation_seed'],
                                                                payoff=contract_data_dict['payoff'],
                                                                tf_option=simulation_dict['tf_generation_option'])
                            
            discounted_payoff = discounted_payoff - discounted_payoff_base

        pathwise_derivs = None
    else:
        raise ValueError("Invalid value for tf_option.")

    # Prepare the return dictionary
    return_dict = {
        'base_scenario': base_scenario,
        'sim_scenario_shifts': sim_scenario_shifts,
        'sim_scenario_levels': sim_scenario_levels,
        'payoff': discounted_payoff,
        'pathwise_derivs': pathwise_derivs,
        'hist_scenario_levels': hist_scenario_levels}

    return return_dict
