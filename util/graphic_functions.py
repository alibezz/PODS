import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from constants import *
import numpy as np


def define_color(antecedent_values, consequent_values, outlier_threshold):
  colors = []
  for value_x, value_y in zip(antecedent_values, consequent_values):
    if (np.fabs(value_x) > outlier_threshold) and (np.fabs(value_y) > outlier_threshold):
      colors.append('red')
    else:
      colors.append('green')
  return colors

def define_size(antecedent_values, consequent_values, outlier_threshold):
  sizes = []
  for value_x, value_y in zip(antecedent_values, consequent_values):
    if (np.fabs(value_x) > 3) and (np.fabs(value_y) > 3):
      sizes.append(200)
    else:
      sizes.append(50)
  return sizes

def define_marker(antecedent_values, consequent_values, outlier_threshold):
  markers = []
  for value_x, value_y in zip(antecedent_values, consequent_values):
    if (np.fabs(value_x) > outlier_threshold) and (np.fabs(value_y) > outlier_threshold):
      markers.append('^')
    else:
      markers.append('o')
  return markers

def plot_aligned_points(antecedent, consequent, thresholds, regions, antec_name, conseq_name, model=None, model_label=''):
  outlier_threshold = thresholds['outlier'][MIN_POSITIVE_ID]
  irregular_threshold = thresholds['irregular'][MIN_POSITIVE_ID]
  all_markers = define_marker(antecedent, consequent, outlier_threshold, irregular_threshold, regions)
  all_colors = define_color(antecedent, consequent, outlier_threshold, irregular_threshold, regions)
  all_sizes = define_size(antecedent, consequent, outlier_threshold)
  for antec, conseq, color_, size_, marker_ in zip(antecedent, consequent, all_colors, all_sizes, all_markers):
    plt.scatter(antec, conseq, color=color_, s=size_, marker=marker_, alpha=0.8)
  if model:
    plt.plot(antecedent, model['params'][1] * np.array(antecedent) + model['params'][0], '-', color='black', linewidth=1.5, label=model_label)
  plt.xlabel(antec_name, fontsize=24)
  plt.ylabel(conseq_name, fontsize=24)
  ax = plt.gca()
  ax.tick_params(axis = 'both', which = 'major', labelsize = 26)
  plt.legend(loc='best', fontsize=16)
  plt.tight_layout()
  plt.savefig(antec_name + '_' + conseq_name + '_' + str(regions) + '_regions.png', dpi=300)
  plt.close()

# def plot_scatterplot(antecedent, consequent, thresholds, antec_name, conseq_name, wls_model, ols_model, wls_model_label, ols_model_label):
#   outlier_threshold = thresholds['outlier'][MIN_POSITIVE_ID]
#   all_markers = define_marker(antecedent, consequent, outlier_threshold)
#   all_colors = define_color(antecedent, consequent, outlier_threshold)
#   all_sizes = define_size(antecedent, consequent, outlier_threshold)
#   for antec, conseq, color_, size_, marker_ in zip(antecedent, consequent, all_colors, all_sizes, all_markers):
#     plt.scatter(antec, conseq, color=color_, s=size_, marker=marker_, alpha=0.8)
#   plt.plot(antecedent, wls_model['params'][1] * np.array(antecedent) + wls_model['params'][0], '-', color='blue', linewidth=1.5, label=wls_model_label)
#   plt.plot(antecedent, ols_model['params'][1] * np.array(antecedent) + ols_model['params'][0], '--', dashes=[5, 10], color='blue', linewidth=1.5, label=ols_model_label)
#   plt.xlabel(r'Temperature', fontsize=24)
#   plt.ylabel(r'Heating complaints', fontsize=24)
#   plt.xlim(-6, 5)
#   plt.ylim(-12, 7)
#   ax = plt.gca()
#   ax.tick_params(axis = 'both', which = 'major', labelsize = 26)
#   plt.legend(loc='best', fontsize=16)
#   plt.tight_layout()
#   plt.savefig(antec_name + '_' + conseq_name + '.png', dpi=600)
#   plt.close()


def plot_histogram_of_values(values, image_name):
  n, bins = np.histogram(values, 10)
  left = np.array(bins[:-1])
  right = np.array(bins[1:])
  bottom = np.zeros(len(left))
  top = bottom + n
  XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T
  barpath = path.Path.make_compound_path_from_polys(XY)
  patch = patches.PathPatch(barpath)
  fig, ax = plt.subplots()
  ax.add_patch(patch)
  ax.set_xlim(left[0], right[-1])
  ax.set_xlabel('Euclidian distance from values to predictions')
  ax.set_ylabel('Number of points')
  ax.set_ylim(bottom.min(), top.max())
  plt.savefig(image_name)
  plt.close()
