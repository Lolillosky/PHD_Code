
from enum import Enum

class MktRisk_Scenarios_Generation_Option(Enum):
    EXPONENTIAL = 1
    LINEAR = 2

class Include_Tensorflow_Calcs_option(Enum):
    YES = 1
    NO = 2

class Simulate_Var_Red_Payoff(Enum):
    YES = 1
    NO = 2


class Base_Scenario_Adj_Option(Enum):
    NPV = 1
    NO = 2
    NPV_PLUS_SENS = 3


class Plat_Analysis_Option(Enum):
    NAIVE = 1
    VAR_REDUCTION = 2
    CONVEXITY = 3
    
    