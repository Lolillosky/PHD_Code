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

