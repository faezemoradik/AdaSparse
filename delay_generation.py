import numpy as np
from scipy.stats import beta


#--------------------------------------------------------------
def PerCoordinateDelay(num_of_clients, max_delay, distribution = 'uniform', alpha=0.8):
  if distribution == 'uniform':
    min_val = 0.0
    max_val = 1.0
    d = np.random.uniform(min_val, max_val, num_of_clients)  

  elif distribution == 'non-uniform':
    d = beta.rvs(alpha, alpha, size=num_of_clients)
    
  return max_delay*d
#--------------------------------------------------------------