import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from metrics import *
from constants import *

def compute_bootstraps_of_adjusted_rsquared(antecedents, consequent, number_of_antecedents, number_of_bootstrap_samples=1000):
  lr = LinearRegression()
  bootstraps = []
  for i in range(0, number_of_bootstrap_samples):
    sample_index = np.random.choice(range(0, len(consequent)), len(consequent))
    x = antecedents[sample_index]
    y = consequent[sample_index]    
    lr.fit(x, y)
    rsquared = lr.score(x, y)
    bootstrap_adjusted_rsquared = compute_adjusted_rsquared(number_of_antecedents, rsquared, len(y))
    bootstraps.append(bootstrap_adjusted_rsquared)
  return sorted(bootstraps)

def compute_confidence_interval_for_adjusted_rsquared(antecedents, consequent, number_of_antecedents):
  total_bootstraps = []
  for i in range(NUMBER_OF_BOOTSTRAP_RUNS):
    total_bootstraps.append(compute_bootstraps_of_adjusted_rsquared(antecedents, consequent, number_of_antecedents, number_of_bootstrap_samples=500))
  low = []
  high = []
  for bootstraps in total_bootstraps:
    low.append(np.mean([bootstraps[9], bootstraps[14]]))
    high.append(np.mean([bootstraps[484], bootstraps[489]]))
  return np.mean(low), np.mean(high)

def compute_zscore_based_confidence_interval_for_adjusted_rsquared(antecedents, consequent, number_of_antecedents):
  q_averages = []
  q_stds = []
  for i in range(NUMBER_OF_BOOTSTRAP_RUNS):
    bootstraps = compute_bootstraps_of_adjusted_rsquared(antecedents, consequent, number_of_antecedents, number_of_bootstrap_samples=500)
    q_averages.append(np.mean(bootstraps))
    q_stds.append(np.std(bootstraps)/np.sqrt(NUMBER_OF_BOOTSTRAP_RUNS))
  upper = np.mean(q_averages) + 3 * np.mean(q_stds)   
  lower = np.mean(q_averages) - 3 * np.mean(q_stds)
  return lower, upper

def plot_histogram(bootstrap_distribution):
  plt.hist(sorted(distribution))
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.title('Distribution of bootstrap-adjusted-rsquared - sample-adjusted-rsquared')
  plt.savefig('bootstrap-distribution' + str(np.random.choice(100000)) + '.png')
  plt.close()

    
