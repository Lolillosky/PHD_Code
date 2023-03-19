import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import tensorflow as tf

def FutureTF(Spot: float, Strike: float, TTM: float, rate: float, div: float) -> float:
    """
    Calculate the theoretical price of a future contract using the cost-of-carry model.

    Args:
        Spot (float): Spot price of the underlying asset
        Strike (float): Strike price of the future contract
        TTM (float): Time-to-maturity of the future contract in years
        rate (float): Risk-free interest rate
        div (float): Dividend yield of the underlying asset

    Returns:
        float: Theoretical price of the future contract
    """
    # Calculate the cost-of-carry formula for future contracts
    # Spot * e^(-div * TTM) - Strike * e^(-rate * TTM)
    return Spot * tf.exp(-div * TTM) - Strike * tf.exp(-rate * TTM)

def BlackScholesTF(Spot: float, Strike: float, TTM: float, rate: float, div: float, Vol: float, IsCall: bool) -> float:
    """
    Calculate the theoretical price of an option contract using the Black-Scholes formula.

    Args:
        Spot (float): Spot price of the underlying asset
        Strike (float): Strike price of the option contract
        TTM (float): Time-to-maturity of the option contract in years
        rate (float): Risk-free interest rate
        div (float): Dividend yield of the underlying asset
        Vol (float): Volatility of the underlying asset
        IsCall (bool): True for call option, False for put option

    Returns:
        float: Theoretical price of the option contract
    """
    # Calculate the forward price of the underlying asset
    Forward = Spot * tf.exp((rate - div) * TTM)

    # Calculate the theoretical price of the option contract using the Black-Scholes formula
    # Call option: Forward * N(d1) - Strike * e^(-rate * TTM) * N(d2)
    # Put option: Strike * e^(-rate * TTM) * N(-d2) - Forward * N(-d1)
    # where d1 = (ln(Forward/Strike) + 0.5 * Vol^2 * TTM) / (Vol * sqrt(TTM))
    # and d2 = d1 - Vol * sqrt(TTM)
    return BlackTF(Forward, Strike, TTM, rate, Vol, IsCall)


def BlackTF(Forward, Strike, TTM, rate, Vol, IsCall):

  '''
  Inputs:
  -------
    Forward (float): Forward value 
    Strike (float): strike price
    TTM (float): time to maturity in years
    rate (float): risk free rate 
    div (float): dividend yield
    Vol (float): volatility
    IsCall (bool): True if call option, False if put option
  Outputs:
  --------
    Option premium
  '''
  dist = tfp.distributions.Normal(loc=0., scale=1.)

  if TTM >0:

    d1 = (tf.math.log(Forward/Strike) + (Vol*Vol/2)*TTM)/(Vol*tf.math.sqrt(TTM))
    d2 = (tf.math.log(Forward/Strike) + (- Vol*Vol/2)*TTM)/(Vol*tf.math.sqrt(TTM))

    if IsCall:     
      return (Forward*dist.cdf(d1)-Strike*dist.cdf(d2))*np.exp(-rate*TTM)
    else:
      return (-Forward*dist.cdf(-d1)+Strike*dist.cdf(-d2))*np.exp(-rate*TTM)
    
  else:
    if IsCall:
      return tf.maximum(Forward-Strike,0)
    else:
      return tf.maximum(-Forward+Strike,0)


def BasketOptionVectorized(num_assets, initial_prices, strike_prices,
                  gross_return_strike,
                  repo_rates, discount_rate, dividends, 
                  volatilities, correlations,
                  time_to_maturity, IsCall):
  '''
   Inputs:
  -------
    num_assets: Number of underlying assets
    initial_prices: Prices of the basket constituents as of value date
    strike_prices: Prices in the denominator of the prod function
    gross_return_strike: gross return strike
    repo_rates: Array of repo rates
    discount_rate: Discount rate
    dividends: Array of dividend yields
    volatilities: Array of volatilities
    correlations: Matrix of instantaneous correlations
    time_to_maturity: Time to maturity
  Output:
    tf.Tensor: Option premium
  '''

  A = tf.pow(tf.reduce_prod(initial_prices/strike_prices, axis = 1), 1.0/num_assets)

  mu = tf.reduce_mean(repo_rates - dividends-0.5*volatilities*volatilities, axis = 1)*time_to_maturity

  Sigma = tf.sqrt(time_to_maturity * tf.reduce_sum(volatilities* 
                  tf.matmul(volatilities, correlations), axis = 1)) / num_assets
                  
  return BlackTF(A*tf.exp(mu+0.5*Sigma*Sigma), gross_return_strike, time_to_maturity, 
                 discount_rate, Sigma/tf.sqrt(time_to_maturity),IsCall)   


class Basket:
    
    def __init__(self, basket_elements, exotic):
        """
        Constructor for Basket class.
        
        Args:
        - basket_elements (list of functions): List of functions that calculate the value of each basket element.
        - exotic (function): Function that calculates the value of the exotic option.
        """
        self.basket_elements = basket_elements
        self.exotic = exotic
        self.hedge_computed = False  # flag variable to track if the hedge has been computed
    
    def compute_hedge(self, base_scenario):
        """
        Computes the hedge weights for the basket and sets them as an attribute of the object.
        
        Args:
        - base_scenario (ndarray): 1D array representing the base scenario for the computation of the hedge.
        """
        base_scenario_tensor = tf.constant(base_scenario.reshape(1,-1))

        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(base_scenario_tensor)
            
            basket_calc = [b(base_scenario_tensor) for b in self.basket_elements]
            exotic_calc = self.exotic(base_scenario_tensor) 

        jac = np.concatenate([tape.gradient(b,base_scenario_tensor).numpy() for b in basket_calc], axis = 0)
        
        grad = tape.gradient(exotic_calc, base_scenario_tensor).numpy().T
        
        self.weights = -np.matmul(np.linalg.pinv(jac),grad).T
        self.hedge_computed = True  # set the flag variable to True
    
    def value_basket(self, scenario):
        """
        Calculates the value of the basket for a given scenario.
        
        Args:
        - scenario (ndarray): 1D or 2D array representing the scenario(s) for which to calculate the basket value.
        
        Returns:
        - ndarray: 1D array representing the basket value(s) for the given scenario(s).
        """
        if not self.hedge_computed:  # check if the hedge has been computed
            raise ValueError("Hedge has not been computed yet")
        
        if scenario.ndim == 1:
            scenario = scenario.reshape(1,-1)
            
        return np.sum(self.weights*np.concatenate([b(scenario).reshape(-1,1) for b in self.basket_elements], axis = 1), axis = 1)



