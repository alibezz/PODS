import time
import datetime
from os import path
import numpy as np

def timing(f):
  def wrap(*args):
    time1 = time.time()
    ret = f(*args)
    time2 = time.time()
    print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
    return ret
  return wrap

# def get_names(combination, dataset_dict):
#   names = [dataset_dict[filename] for filename in combination]
#   return ", ".join(names)

# def intersect(d, ignore=[]):
#   tmp = d
#   for elem in ignore:
#     del tmp[elem]
#   if tmp:
#     return set.intersection(*map(set,tmp))
#   return tmp

def get_radical(string):
  #example: metric_granularity_4-day_311_BLOCKED_DRIVEWAY_2011_to_2013.txt => BLOCKED_DRIVEWAY
  #return '_'.join(path.basename(string).split('.')[0].split('window_')[1].split('_stds')[1:])
  return string.split('311_')[1].split('_2011')[0]

def format_date_particle(particle):
  if particle < 10:
    return '0' + str(particle)
  return str(particle)

def get_posterior_timestamps(timestamp, number_of_approximate_matches):
  # get posterior timestamps in daily granularity
  # format of timestamp: 2011-08-25 20:00
  
  date, hour = timestamp.split()
  year, month, day = date.split('-')
  date = datetime.datetime(int(year), int(month), int(day))
  posterior_timestamps = []
  for i in range(number_of_approximate_matches):
    new_date = date + datetime.timedelta(days=i+1)
    posterior_timestamps.append('-'.join([str(new_date.year), format_date_particle(new_date.month), format_date_particle(new_date.day)]) + ' ' + hour)
  return posterior_timestamps

def update_buckets(value, buckets):
  if np.fabs(value) < 0.5:
    key = 0.0
  elif np.fabs(value) < 1.0:
    key = 0.5
  elif np.fabs(value) < 1.5:
    key = 1.0
  elif np.fabs(value) < 2.0:
    key = 1.5
  elif np.fabs(value) < 2.5:
    key = 2.0
  elif np.fabs(value) < 3.0:
    key = 2.5
  else:
    key = 3.0
  try:
    buckets[key] += 1
  except:
    buckets[key] = 1
  return buckets
