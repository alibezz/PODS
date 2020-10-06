import numpy as np
from constants import *
from util.misc import update_buckets

#TODO refactor and simplify

def valid_date(date, dates): 
  return (not dates) or (date in dates)

def valid_value(value, min_positive, max_positive, min_negative, max_negative):
  try:
    return (float(value) > min_positive and float(value) < max_positive) or (float(value) < max_negative and float(value) > min_negative) 
  except:
    return False #in case value is not a proper numerical value, like NA

def get_data(filename, thrs=[0.0, float('inf'), float('-inf'), 0.0], dates=[]):
  #thrs: min_positive, max_negative. A value has to be > min_positive or < max_negative to be valid
  #dates: if different from [], indicates which dates we want to keep; if it's equals to [], all dates are kept
  
  min_positive = thrs[MIN_POSITIVE_ID]; max_positive = thrs[MAX_POSITIVE_ID]
  min_negative = thrs[MIN_NEGATIVE_ID]; max_negative = thrs[MAX_NEGATIVE_ID]  
  lines = open(filename, 'r').readlines()[1:]
  range_values = {}
  for l in lines:
    date, value = l.strip().split(SEPARATOR)
    if valid_date(date, dates):
      if valid_value(value, min_positive, max_positive, min_negative, max_negative):
        range_values[date] = float(value)
  return range_values

def determine_scores(cumulative_score, zscore):
  if cumulative_score == float('-inf'):
    cumulative_score = zscore
  else:
    cumulative_score = LAMBDA * zscore + (1 - LAMBDA) * cumulative_score
  if max(np.fabs(zscore), np.fabs(cumulative_score)) == np.fabs(zscore):
    final_score = zscore
  else:
    final_score = cumulative_score
  return final_score, cumulative_score

def get_data_list_format(filename, thrs=[0.0, float('inf'), float('-inf'), 0.0], dates=[]):
  '''
  The list format is important for when the order of the days matter, e.g., for an
  "approximately temporal" index
  '''

  min_positive = thrs[MIN_POSITIVE_ID]; max_positive = thrs[MAX_POSITIVE_ID]
  min_negative = thrs[MIN_NEGATIVE_ID]; max_negative = thrs[MAX_NEGATIVE_ID]  
  lines = open(filename, 'r').readlines()[1:]
  range_values = []
  cumulative_score = float('-inf')
  zscore_buckets = {}
  final_score_buckets = {}
  for l in lines:
    date, value = l.strip().split(SEPARATOR)
    if valid_value(value, min_positive, max_positive, min_negative, max_negative):
      final_score, cumulative_score = determine_scores(cumulative_score, float(value))
      range_values.append((date, final_score))
  return range_values

def get_zscores(filename, time_window_size):
  '''
  to make it iterative, especially wrt standard deviation, we use 
  the corrected sample variance, i.e.,
  \sum(X_i - mean)^2 = \sum(X_i^2) - n(mean^2)
  '''
  tuples = [l.strip().split('|') for l in open(filename, 'r').readlines()[1:]]
  init_window = 0
  init_value = np.nan
  count = time_window_size
  zscore_tuples = []
  while init_window + time_window_size < len(tuples):
    if np.isnan(init_value): #compute first stats for mean and standard deviation
      init_value = np.log(float(tuples[init_window][1]) + 1.)
      sum_X = np.sum([np.log(float(i[1])+1.) for i in tuples[init_window:init_window+time_window_size]])
      sum_X_2 = np.sum([np.log(float(i[1])+1.) ** 2 for i in tuples[init_window:init_window+time_window_size]])
    else: #update stats for mean and standard deviation
      init_value = np.log(float(tuples[init_window-1][1]) + 1.) 
      new_end_value = np.log(float(tuples[init_window+time_window_size-1][1]) + 1.)
      sum_X -= init_value
      sum_X += new_end_value
      sum_X_2 -= init_value ** 2
      sum_X_2 += new_end_value ** 2    
    mean = sum_X/count
    std_numerator = sum_X_2 - count*(mean ** 2)
    std = np.sqrt(std_numerator/count)
    #print 'iterative mean', mean, 'numpy mean', np.mean([np.log(float(i[1])+1.) for i in tuples[init_window:init_window+time_window_size]])
    #print 'iterative std', std, 'numpy std', np.std([np.log(float(i[1])+1.) for i in tuples[init_window:init_window+time_window_size]])
    date, value = tuples[init_window+time_window_size]
    zscore_tuples.append((date, (np.log(float(value)+1.) - mean)/std))
    init_window += 1
  return zscore_tuples

def get_normalized_data_list_format(filename, time_window, thrs=[0.0, float('inf'), float('-inf'), 0.0], dates=[]):
  dates_zscores = get_zscores(filename, time_window)
  min_positive = thrs[MIN_POSITIVE_ID]; max_positive = thrs[MAX_POSITIVE_ID]
  min_negative = thrs[MIN_NEGATIVE_ID]; max_negative = thrs[MAX_NEGATIVE_ID]  
  range_values = []
  cumulative_score = float('-inf')
  zscore_buckets = {}
  final_score_buckets = {}
  for tuple in dates_zscores:
    date = tuple[0]
    value = tuple[1]
    if valid_value(value, min_positive, max_positive, min_negative, max_negative):
      final_score, cumulative_score = determine_scores(cumulative_score, float(value))
      range_values.append((date, final_score))
      #range_values.append((date, float(value)))
  return range_values
  
def get_file_data_list_format(filename, thrs=[0.0, float('inf'), float('-inf'), 0.0], dates=[]):
  min_positive = thrs[MIN_POSITIVE_ID]; max_positive = thrs[MAX_POSITIVE_ID]
  min_negative = thrs[MIN_NEGATIVE_ID]; max_negative = thrs[MAX_NEGATIVE_ID]  
  range_values = []
  tuples = [l.strip().split('|') for l in open(filename, 'r').readlines()[1:]]
  for tuple in tuples:
    date = tuple[0]
    value = tuple[1]
    if valid_value(value, min_positive, max_positive, min_negative, max_negative):
      range_values.append((date, float(value)))
  return range_values

def is_valid(value, thrs):
  min_positive = thrs[MIN_POSITIVE_ID]; max_positive = thrs[MAX_POSITIVE_ID]
  min_negative = thrs[MIN_NEGATIVE_ID]; max_negative = thrs[MAX_NEGATIVE_ID]  
  return valid_value(value, min_positive, max_positive, min_negative, max_negative)

def value_is_outlier(value, outlier_thresholds):
  return is_valid(value, outlier_thresholds)

def get_point_type(antec, conseq, irregular_thresholds, outlier_thresholds):
  if is_valid(antec, outlier_thresholds) and is_valid(conseq, outlier_thresholds):
    return 'outlier'
  if (is_valid(antec, irregular_thresholds) and is_valid(conseq, irregular_thresholds)) or (is_valid(antec, irregular_thresholds) and is_valid(conseq, outlier_thresholds)) or (is_valid(antec, outlier_thresholds) and is_valid(conseq, irregular_thresholds)):
    return 'irregular'
  return 'normal'

def get_aligned_datasets_dp_features(antecedent_data, consequent_data, antecedent_outliers, consequent_outliers):
  final_dates = list(set(antecedent_data.keys()) & set(consequent_data.keys()))
  normal_points = []; outlier_points = []
  for date in final_dates:
    if antecedent_data[date] in antecedent_outliers and consequent_data[date] in consequent_outliers:
      outlier_points.append((antecedent_data[date], consequent_data[date]))
    else:
      normal_points.append((antecedent_data[date], consequent_data[date]))    
  return {'normal': normal_points, 'outlier': outlier_points}

  return
  
def get_aligned_datasets(antecedent_data, consequent_data, outlier_thresholds, irregular_thresholds):
  '''
  this function aligns the datasets temporally (only dates in common)
  '''

  final_dates = list(set(antecedent_data.keys()) & set(consequent_data.keys()))
  normal_points = []; irregular_points = []; outlier_points = []
  for date in final_dates:
    point_type = get_point_type(antecedent_data[date], consequent_data[date], irregular_thresholds, outlier_thresholds)
    if point_type == 'outlier':
      outlier_points.append((antecedent_data[date], consequent_data[date]))
    elif point_type == 'irregular':
      irregular_points.append((antecedent_data[date], consequent_data[date]))
    else:
      normal_points.append((antecedent_data[date], consequent_data[date]))    
  return {'normal': normal_points, 'irregular': irregular_points, 'outlier': outlier_points}
