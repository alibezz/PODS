import numpy as np
from scipy.stats import ttest_ind, t, sem
from constants import *

def compute_adjusted_rsquared(number_of_antecedents, rsquared, number_of_points):
  num = (1.0 - rsquared) * (number_of_points - 1.0)
  den = number_of_points - number_of_antecedents - 1.0
  return 1.0 - (num/den)

def compute_rsquared(weights, y_values, predictions):
  # the equations below work for the weighted and the unweighted cases
  #num = summation[w_i * (y_i - y"_i)^2]
  #den = summation[(w_i * (y_i - y'_w)^2]
  #r-squared = 1 - num/den
  weights = np.array(weights)
  y_values = np.array(y_values)
  predictions = np.array(predictions)
  num = np.sum(weights * (y_values - predictions) ** 2.0)
  den = np.sum(weights * (y_values - np.average(y_values, weights=weights)) ** 2.0)
  return 1.0 - num/den 

def compute_average_euclidian_distance(y_values, predictions):
  return np.linalg.norm(np.array(y_values) - np.array(predictions))/len(predictions)

def means_are_statistically_equivalent(distance_distr1, distance_distr2, p_value_threshold):
  #the null hypothesis is that the means of these distributions are the same
  #this function returns True if we can't reject the null hypothesis
  t, p = ttest_ind(distance_distr1, distance_distr2, equal_var=False)
  return p > p_value_threshold
  
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), sem(data)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_bootstrapped_upper_bound_percentile(distance_distr, percentile=95):
  percentiles = []
  for i in xrange(NUMBER_OF_BOOTSTRAP_SAMPLES):
    sample = np.random.choice(distance_distr, size=BOOTSTRAP_SAMPLE_SIZE, replace=True)
    percentiles.append(np.percentile(sample, percentile))
  mean, lower_bound, upper_bound = mean_confidence_interval(percentiles)
  #print 'mean of percentiles =', mean, 'lower bound', lower_bound, 'upper bound', upper_bound
  return upper_bound

