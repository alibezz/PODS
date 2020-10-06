# -*- coding: utf-8 -*-

'''
This program gets a directory with NORMALIZED attributes aggregated by day, hour, or month and runs TiPO for all distinct pairs. 
Currently tested for two types of normalization: mean residuals and cumulative scores. However, it should work for any normalization
that outputs values that are symmetric around a certain mean
argv[1] => directory
argv[2] => a file with the thresholds that have to be used to know the total of near-outlier/outlier points in all normalized attributes
argv[3] => a real number  indicating the base alpha for the weighting scheme
argv[4] => valid outlier percentage (e.g., 0.67), i.e., the percentage of outliers whose distance wrt to their predictions is bounded by the 95th percentile of the resampled model
argv[5] => minimum adjusted rsquared above which the regression of a model is considered to fit the data adequately

Example of execution: python find_good_correlations_from_normalized_data.py folder_with_files thresholds.txt 0.5 0.95 0.4
'''

import sys
from file_parser import *
from detect_trend import run_cached_normalized_tipo

def detect_meaningful_pairs(directory, threshold_file, base_wls_parameter, valid_outlier_percentage, min_adj_rsquared):
  thresholds = parse_thresholds(threshold_file)
  run_cached_normalized_tipo(directory, thresholds, base_wls_parameter, valid_outlier_percentage, min_adj_rsquared)
  
if __name__=='__main__':
  directory = sys.argv[1]
  threshold_file = sys.argv[2]
  base_wls_parameter = float(sys.argv[3])
  valid_outlier_percentage = float(sys.argv[4])
  min_adj_rsquared = float(sys.argv[5])
  detect_meaningful_pairs(directory, threshold_file, base_wls_parameter, valid_outlier_percentage, min_adj_rsquared) 
  
