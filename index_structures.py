'''
This script contains all functions related to the temporal index
'''
from os import listdir
from os.path import isfile, join
from data_ingestion import *
from itertools import combinations
from util.misc import *

def create_outlier_list_per_file(directory, time_windows):
    return {join(directory, f + '_' + str(tw)): [] for f in listdir(directory) for tw in time_windows if isfile(join(directory, f))}

def create_outlier_list_per_normalized_file(directory):
    return {join(directory, f): [] for f in listdir(directory) if isfile(join(directory, f))}

#@timing
def retrieve_files_with_co_occ_outliers(files_data, temporal_index):
    distinct_file_pairs = set()
    for timestamp in temporal_index.keys():
        if len(temporal_index[timestamp]) > 1:
            filenames_in_timestamp = sorted(temporal_index[timestamp].keys())
            #print filenames_in_timestamp
            pairs = combinations(filenames_in_timestamp, 2)
            for p in pairs:
                distinct_file_pairs.add(p)
    total_content = []
    for p in distinct_file_pairs:
        pair_content = ((p[0], dict(files_data[p[0]])), (p[1], dict(files_data[p[1]])))
        total_content.append(pair_content)
    return total_content

def retrieve_pairs_of_files_with_co_occ_outliers(temporal_index):
    distinct_file_pairs = set()
    for timestamp in temporal_index.keys():
        if len(temporal_index[timestamp]) > 1:
            filenames_in_timestamp = sorted(temporal_index[timestamp])
            pairs = combinations(filenames_in_timestamp, 2)
            for p in pairs:
                distinct_file_pairs.add(p)
    return distinct_file_pairs

def exact_matching(temporal_index, timestamp, index_str, current_outlier_index):
    if temporal_index.has_key(timestamp):
        if temporal_index[timestamp].has_key(index_str):
            temporal_index[timestamp][index_str].append(current_outlier_index)
        else:
            temporal_index[timestamp][index_str] = [current_outlier_index] 
    else:
        temporal_index[timestamp] = {index_str: [current_outlier_index]}
    return temporal_index

def index_raw_files(directory, time_windows, outlier_thresholds):
    outlier_lists = create_outlier_list_per_file(directory, time_windows)
    data = {}
    temporal_index = {}
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    for tw in time_windows:
        for filename in files:
            index_str = filename + '_' + str(tw)
            data[index_str] = get_normalized_data_list_format(filename, tw)
            outliers_in_file = [entry for entry in data[index_str] if value_is_outlier(entry[1], outlier_thresholds)]
            for (timestamp, outlier_value) in outliers_in_file:
                outlier_lists[index_str].append((timestamp, outlier_value))
                current_outlier_index = len(outlier_lists[index_str])-1
                temporal_index = exact_matching(temporal_index, timestamp, index_str, current_outlier_index)
    return data, temporal_index

def populate_index(temporal_index, timestamp, filename):
    if temporal_index.has_key(timestamp):
        temporal_index[timestamp].append(filename) 
    else:
        temporal_index[timestamp] = [filename]
    return temporal_index

#@timing
def index_normalized_files(directory, outlier_thresholds):
    outlier_lists = create_outlier_list_per_normalized_file(directory)
    data = {}
    temporal_index = {}
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    for filename in files:
        #assumes that dominant scores are already computed in filename
        data[filename] = get_file_data_list_format(filename)
        outliers_in_file = [entry for entry in data[filename] if value_is_outlier(entry[1], outlier_thresholds)]
        for (timestamp, outlier_value) in outliers_in_file:
            outlier_lists[filename].append((timestamp, outlier_value))
            temporal_index = populate_index(temporal_index, timestamp, filename)
    return data, temporal_index
