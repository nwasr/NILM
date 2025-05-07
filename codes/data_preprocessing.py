import numpy as np
from math import ceil

def train_test_split(time_series, train_end_timestamp):

  train = time_series[:train_end_timestamp].values
  test = time_series[train_end_timestamp:].values

  return train, test

def train_val_test_split(time_series, train_start_timestamp, train_end_timestamp, val_end_timestamp):

  train = time_series[train_start_timestamp:train_end_timestamp].values
  val = time_series[train_end_timestamp:val_end_timestamp].values
  test = time_series[val_end_timestamp:].values

  return train, val, test

def normalize_data(data, min_value=0.0, max_value=1.0):

  data -= min_value
  data /= max_value - min_value
  return data

def standardize_data(data, mu=0.0, sigma=1.0):

  data -= mu
  data /= sigma
  return data

def zero_pad(data, window_size):

  pad_width = ceil(window_size / 2)
  padded = np.pad(data, (pad_width, pad_width), 'constant', constant_values=(0,0))
  return padded
