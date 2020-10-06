'''
Given a file with initial scores, this code computes the corresponding cumulative scores 
'''

import sys
import numpy as np
from constants import *

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

if __name__ == '__main__':

  initial_filename = sys.argv[1]
  cumulative_score = float('-inf')
  
  with open(initial_filename, 'r') as initial_file:
      final_file = open('dominant_scores.csv', 'w')
      for line in initial_file:
          date, value = line.strip().split(SEPARATOR)
          try:
            final_score, cumulative_score = determine_scores(cumulative_score, float(value))
            final_file.write(date + SEPARATOR + str(final_score) + '\n')
          except:
            continue
