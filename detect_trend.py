'''
This file contains all functions related to trend detection -- the core involves checking to which extent 
one could see a trend coming from areas with lower mean residuals 
'''

#TODO document all functions

#TODO review what I am importing, refactor and simplify

from index_structures import *
from solutions import *
from metrics import *
from constants import *
from data_ingestion import *
from file_parser import *
from util.misc import *
from util.graphic_functions import *
import numpy as np
from os.path import isfile, join
from itertools import combinations
from os import listdir
import random
from scipy.stats import pearsonr

def check_if_could_see_it_coming(inlier_error, total_error, outlier_error, regression_coefficients, regression_coefficient_pvalues, adj_rsquared, 
                                 model_distance_distr, result_distance_distr, valid_outlier_percentage, min_adj_rsquared):
  valid_slope_pvalue = regression_coefficient_pvalues[SLOPE_VALUE_ID] <= P_VALUE_THRESHOLD
  upper_bound_mean_percentiles = get_bootstrapped_upper_bound_percentile(model_distance_distr)
  number_of_outliers_that_meet_percentile_criterion = 1. * len([o_dist for o_dist in result_distance_distr if o_dist <= upper_bound_mean_percentiles])
  if np.isnan(inlier_error) or np.isinf(inlier_error) or np.isnan(total_error) or np.isinf(total_error) or len(result_distance_distr) == 0:
    return valid_slope_pvalue, False
  return valid_slope_pvalue, number_of_outliers_that_meet_percentile_criterion/len(result_distance_distr) >= valid_outlier_percentage and adj_rsquared >= min_adj_rsquared
  
def prediction_results(model, outlier_antecedent, outlier_consequent, antecedent, consequent):
  results = {}
  results['predictions'] = model['params'][1] * np.array(outlier_antecedent) + model['params'][0]
  results['total_weights'] = np.append(model['weights'], [1.0 for i in outlier_consequent])
  results['total_predictions'] = np.append(model['predictions'], results['predictions'])
  results['error'] = compute_average_euclidian_distance(consequent, results['total_predictions'])
  distance_distribution = [np.fabs(i) for i in np.array(consequent) - np.array(results['total_predictions'])]
  outlier_distance_distribution = [np.fabs(i) for i in np.array(outlier_consequent) - np.array(results['predictions'])]
  results['distance_distribution'] = distance_distribution
  results['outlier_distance_distribution'] = outlier_distance_distribution
  mean, std = np.mean(model['distance_distribution']), np.std(model['distance_distribution'])
  return results

#@timing
def compute_regressions(antecedent_name, consequent_name, data, alpha, thresholds, valid_outlier_percentage, min_adj_rsquared):
  #compute models
  outlier_antecedent = [i[0] for i in data['outlier']]
  outlier_consequent = [i[1] for i in data['outlier']]
  normal_irregular_antecedent = [i[0] for i in data['irregular']] + [i[0] for i in data['normal']]
  normal_irregular_consequent = [i[1] for i in data['irregular']] + [i[1] for i in data['normal']]  
  
  ######## WLS ########
  model_wls_all = wls_all(normal_irregular_antecedent, normal_irregular_consequent, outlier_antecedent, outlier_consequent, alpha, thresholds['outlier'])
  model_wls_all_symmetric = wls_all(normal_irregular_consequent, normal_irregular_antecedent, outlier_consequent, outlier_antecedent, alpha, thresholds['outlier'])
  wls_all_valid_pvalues, wls_all_could_see_it_coming = check_if_could_see_it_coming(model_wls_all['inlier_error'], model_wls_all['error'], model_wls_all['outlier_error'], model_wls_all['params'], model_wls_all['pvalues'], model_wls_all['adj_rsquared'], model_wls_all['inlier_distance_distribution'], model_wls_all['outlier_distance_distribution'], valid_outlier_percentage, min_adj_rsquared)
  wls_all_symmetric_valid_pvalues, wls_all_symmetric_could_see_it_coming = check_if_could_see_it_coming(model_wls_all_symmetric['inlier_error'], model_wls_all_symmetric['error'], model_wls_all_symmetric['outlier_error'],  model_wls_all_symmetric['params'], model_wls_all_symmetric['pvalues'], model_wls_all_symmetric['adj_rsquared'], model_wls_all_symmetric['inlier_distance_distribution'],  model_wls_all_symmetric['outlier_distance_distribution'], valid_outlier_percentage, min_adj_rsquared)
  if (wls_all_could_see_it_coming and wls_all_valid_pvalues) or (wls_all_symmetric_could_see_it_coming and wls_all_symmetric_valid_pvalues):
    print '*** PODS saw it coming for', antecedent_name, 'and', consequent_name
  else:
    print '*** PODS did not see it coming for', antecedent_name, 'and', consequent_name
  
#@timing
def detect_trend(antecedent_name, consequent_name, data, thrs, alpha, valid_outlier_percentage, min_adj_rsquared):
  trend_exists = compute_regressions(antecedent_name, consequent_name, data, alpha, thrs, valid_outlier_percentage, min_adj_rsquared)

#@timing
def run_AA_CD_VoM(antec_data, conseq_data, antecedent_name, consequent_name, thresholds, alpha, valid_outlier_percentage, min_adj_rsquared):
  separated_data = get_aligned_datasets(antec_data, conseq_data, thresholds['outlier'], thresholds['irregular'])
  detect_trend(antecedent_name, consequent_name, separated_data, thresholds, alpha, valid_outlier_percentage, min_adj_rsquared)

def run_tipo_over_dp_features(content_1, content_2, outliers_1, outliers_2, attribute_file_1, attribute_file_2,
                              base_wls_parameter, valid_outlier_percentage, min_adj_rsquared):
  separated_data = get_aligned_datasets_dp_features(content_1, content_2, outliers_1, outliers_2)
  return compute_dp_regressions(attribute_file_1, attribute_file_2, separated_data, base_wls_parameter,
                                valid_outlier_percentage, min_adj_rsquared)
  
#@timing
def run_cached_tipo(directory, thresholds, time_windows, alpha):
  files_data, outlier_temporal_index = index_raw_files(directory, time_windows, thresholds['outlier'])
  pairs = retrieve_files_with_co_occ_outliers(files_data, outlier_temporal_index)
  for p in pairs:
    run_AA_CD_VoM(p[0][1], p[1][1], p[0][0], p[1][0], thresholds, alpha)

#@timing
def run_cached_normalized_tipo(directory, thresholds, alpha, valid_outlier_percentage, min_adj_rsquared):
  files_data, outlier_temporal_index = index_normalized_files(directory, thresholds['outlier'])
  pairs = retrieve_pairs_of_files_with_co_occ_outliers(outlier_temporal_index)
  for p in pairs:
    content_0 = dict(files_data[p[0]])
    content_1 = dict(files_data[p[1]])
    run_AA_CD_VoM(content_0, content_1, p[0], p[1], thresholds, alpha, valid_outlier_percentage, min_adj_rsquared)

#@timing
def run_normalized_tipo_no_index(directory, thresholds, alpha, valid_outlier_percentage, min_adj_rsquared):
  files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
  total_data = {}
  for filename in files:
    total_data[filename] = get_data(filename)
  pairs = list(combinations(total_data.keys(), 2))
  for p in pairs:
    content_0 = dict(total_data[p[0]])
    content_1 = dict(total_data[p[1]])
    run_AA_CD_VoM(content_0, content_1, p[0], p[1], thresholds, alpha, valid_outlier_percentage, min_adj_rsquared)
