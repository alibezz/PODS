import statsmodels.api as sm
from metrics import *
from constants import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def difference(value1, value2):
  if np.absolute(value2) >= np.absolute(value1):
    return 0.0  
  return (np.absolute(value1) - np.absolute(value2))

def combination_minimum(value1, value2):
  return min(value1, value2) ** 2 #important to keep consistency with paper, as weights are squared in optimization

def get_continuous_weights(shaped_antecedent, shaped_consequent, alpha, thresholds):
  '''
  this method does the following:in each dimension, if the value is above the dimension, the weight is going to be the maximum, otherwise we perform a 
  transformation more or less like in get_continuous_weights_y_axis
  '''
  weights = []
  for antec, conseq in zip(shaped_antecedent, shaped_consequent):
    value1 = alpha ** difference(thresholds[MIN_POSITIVE_ID], antec)
    value2 = alpha ** difference(thresholds[MIN_POSITIVE_ID], conseq)
    weights.append(combination_minimum(value1, value2))
    
  return weights

def get_dp_continuous_weights(all_antecedent, all_consequent, alpha, outlier_antecedent, outlier_consequent):
  '''
  this method does the following: for both antecedent and consequent, it gets the minimum positive and negative 
  outlier values. if there are no positive (or negative) outlier values, we set the minimum as inf. for each value in 
  the antecedent or consequent, we check whether it is above the minimum positive (or below the minimum negative). if 
  this is the case, the weight is going to be the maximum, otherwise we perform a transformation 
  '''
  minimum_positive_antec_outlier = minimum_positive_conseq_outlier = max([np.fabs(i) for i in all_antecedent])
  minimum_negative_antec_outlier = minimum_negative_conseq_outlier = max([np.fabs(i) for i in all_consequent])
  for outlier_antec, outlier_conseq in zip(outlier_antecedent, outlier_consequent):
    if outlier_antec > 0 and outlier_antec < minimum_positive_antec_outlier:
      minimum_positive_antec_outlier = outlier_antec
    elif outlier_antec <= 0 and np.fabs(outlier_antec) < np.fabs(minimum_negative_antec_outlier):
      minimum_negative_antec_outlier = outlier_antec
    if outlier_conseq > 0 and outlier_conseq < minimum_positive_conseq_outlier:
      minimum_positive_conseq_outlier = outlier_conseq
    elif outlier_conseq <= 0 and np.fabs(outlier_conseq) < np.fabs(minimum_negative_conseq_outlier):
      minimum_negative_conseq_outlier = outlier_conseq

  weights = []  
  for antec, conseq in zip(all_antecedent, all_consequent):
    if antec > 0: 
      value1 = alpha ** difference(minimum_positive_antec_outlier, antec)
    else:
      value1 = alpha ** difference(minimum_negative_antec_outlier, antec)
    if conseq > 0:
      value2 = alpha ** difference(minimum_positive_conseq_outlier, conseq)
    else:
      value2 = alpha ** difference(minimum_negative_conseq_outlier, conseq)
    weights.append(combination_minimum(value1, value2))
  return weights
  
def generate_model(antecedent, consequent, weights):
  model = sm.WLS(consequent, antecedent, weights=weights)
  results = model.fit()
  predicted = results.predict()
  rsquared = compute_rsquared(weights, consequent, predicted)
  return results.params, predicted, results.pvalues, compute_adjusted_rsquared(1, rsquared, len(consequent))

def model_wls(antecedent, consequent, alpha, outlier_thresholds):
  wls_weights = get_continuous_weights(antecedent, consequent, alpha, outlier_thresholds)
  X = sm.add_constant(antecedent)
  wls_params, wls_predictions, wls_pvalues, wls_adj_rsquared = generate_model(X, consequent, wls_weights)
  wls_error = compute_average_euclidian_distance(consequent, wls_predictions)
  distance_distribution = [np.fabs(i) for i in np.array(consequent) - np.array(wls_predictions)]
  model = {}
  model['weights'] = wls_weights
  model['params'] = wls_params
  model['predictions'] = wls_predictions
  model['pvalues'] = wls_pvalues
  model['adj_rsquared'] = wls_adj_rsquared
  model['error'] = wls_error
  model['distance_distribution'] = distance_distribution
  return model

def wls_all(inlier_antecedent, inlier_consequent, outlier_antecedent, outlier_consequent, alpha, outlier_thresholds):
  all_antecedent = np.array(inlier_antecedent + outlier_antecedent)
  all_consequent = np.array(inlier_consequent + outlier_consequent)
  wls_weights = get_continuous_weights(all_antecedent, all_consequent, alpha, outlier_thresholds)
  X = sm.add_constant(all_antecedent)
  wls_params, wls_predictions, wls_pvalues, wls_adj_rsquared = generate_model(X, all_consequent, wls_weights)
  all_error = compute_average_euclidian_distance(all_consequent, wls_predictions)
  model = {}
  model['weights'] = wls_weights
  model['params'] = wls_params
  model['predictions'] = wls_predictions
  model['pvalues'] = wls_pvalues
  model['adj_rsquared'] = wls_adj_rsquared
  model['error'] = all_error
  try:
    model['inlier_error'] = compute_average_euclidian_distance(inlier_consequent, wls_predictions[:-len(outlier_consequent)])
    model['inlier_distance_distribution'] = [np.fabs(i) for i in np.array(inlier_consequent) - np.array(wls_predictions[:-len(outlier_consequent)])]
    model['outlier_error'] = compute_average_euclidian_distance(outlier_consequent, wls_predictions[len(wls_predictions)-len(outlier_consequent):])
    model['outlier_distance_distribution'] = [np.fabs(i) for i in np.array(outlier_consequent) - np.array(wls_predictions[len(wls_predictions)-len(outlier_consequent):])]
  except ValueError:
    model['inlier_error'] = compute_average_euclidian_distance(inlier_consequent, wls_predictions)
    model['inlier_distance_distribution'] = [np.fabs(i) for i in np.array(inlier_consequent) - np.array(wls_predictions)]
    model['outlier_error'] = []
    model['outlier_distance_distribution'] = []
  return model

def wls_dp_all(inlier_antecedent, inlier_consequent, outlier_antecedent, outlier_consequent, alpha):
  all_antecedent = np.array(inlier_antecedent + outlier_antecedent)
  all_consequent = np.array(inlier_consequent + outlier_consequent)
  wls_weights = get_dp_continuous_weights(all_antecedent, all_consequent, alpha, outlier_antecedent, outlier_consequent)
  X = sm.add_constant(all_antecedent)
  wls_params, wls_predictions, wls_pvalues, wls_adj_rsquared = generate_model(X, all_consequent, wls_weights)
  all_error = compute_average_euclidian_distance(all_consequent, wls_predictions)
  model = {}
  model['weights'] = wls_weights
  model['params'] = wls_params
  model['predictions'] = wls_predictions
  model['pvalues'] = wls_pvalues
  model['adj_rsquared'] = wls_adj_rsquared
  model['error'] = all_error
  try:
    model['inlier_error'] = compute_average_euclidian_distance(inlier_consequent, wls_predictions[:-len(outlier_consequent)])
    model['inlier_distance_distribution'] = [np.fabs(i) for i in np.array(inlier_consequent) - np.array(wls_predictions[:-len(outlier_consequent)])]
    model['outlier_error'] = compute_average_euclidian_distance(outlier_consequent, wls_predictions[len(wls_predictions)-len(outlier_consequent):])
    model['outlier_distance_distribution'] = [np.fabs(i) for i in np.array(outlier_consequent) - np.array(wls_predictions[len(wls_predictions)-len(outlier_consequent):])]
  except ValueError:
    model['inlier_error'] = compute_average_euclidian_distance(inlier_consequent, wls_predictions)
    model['inlier_distance_distribution'] = [np.fabs(i) for i in np.array(inlier_consequent) - np.array(wls_predictions)]
    model['outlier_error'] = []
    model['outlier_distance_distribution'] = []
  return model
