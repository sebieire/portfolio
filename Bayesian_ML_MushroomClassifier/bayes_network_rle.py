"""
Bayesian Network for Residual Life Expectancy (RLE) Analysis
Implementation of probabilistic modeling for health risk factors
"""

from lib_aima.probability import BayesNet, BayesNode, enumeration_ask, likelihood_weighting

run_from_notebook = True

def get_RLE_bayes_network():
    """    
        - Implementation Of A Bayes Network -
        uses predefined values to calculate RLE Loss
        returns the network
    """    
  
    network = BayesNet([
        ('Is_Smoker','', 0.23),
        ('Is_Heavy_Drinker','', 0.0156),
        ('Is_Overweight','', 0.6),
        ('Low_Physical_Acitivity','', 0.3),
        ('Is_Female','', 0.5081),
        
        ('Health_Risk', ['Is_Smoker', 'Is_Heavy_Drinker','Is_Overweight','Low_Physical_Acitivity'],
         {(True,True,True,True): 0.999, 
          (True,True,True,False): 0.898, 
          (True,True,False,True): 0.810,
          (True,True,False,False): 0.708,
          (True,False,True,True): 0.815,
          (True,False,True,False): 0.713,
          (True,False,False,True): 0.625,
          (True,False,False,False): 0.523,
          (False,True,True,True): 0.477,
          (False,True,True,False): 0.375,
          (False,True,False,True): 0.287,
          (False,True,False,False): 0.185,
          (False,False,True,True): 0.292,
          (False,False,True,False): 0.190,
          (False,False,False,True): 0.102,
          (False,False,False,False): 0.001
         }),
        
        ('RLE_Loss', ['Health_Risk', 'Is_Female'],
         {(True,True): 0.7694,
          (True,False): 0.9411,
          (False,True): 0.0647,
          (False,False): 0.0235
         })

        ])
    
    return network


if not run_from_notebook:

    # Main / Run
    if __name__ == "__main__":    
    
    
        network = get_RLE_bayes_network()
        print("Testing - Health Risk Var:\n", network.variable_node('Health_Risk').cpt)

