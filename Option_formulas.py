import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

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

