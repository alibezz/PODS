'''
This part of the code contains methods to parse thresholds and time window sizes
'''

def get_thresholds(line):
  fields = line.strip().split()
  values = []
  for f in fields:
    if f == 'inf':
      values.append(float('inf'))
    elif f == '-inf':
      values.append(float('-inf'))
    else:
      values.append(float(f))
  return values

def parse_thresholds(threshold_file):
  tf = open(threshold_file, 'r')
  thresholds = {}
  thresholds['outlier'] = get_thresholds(tf.readline())
  thresholds['irregular'] = get_thresholds(tf.readline())
  tf.close()
  return thresholds

def parse_outliers(outlier_file):
  '''
  outlier_file has one line in format [x0, ..., xN]
  '''
  outliers = eval(open(outlier_file, 'r').readline().strip())
  return outliers
  
def parse_time_windows(time_window_file):
  window_sizes = [int(i) for i in open(time_window_file, 'r').readline().strip().split()]
  return window_sizes
