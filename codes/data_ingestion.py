import tensorflow as tf
import numpy as np
import math

class DataIngestor(tf.keras.utils.Sequence):

  def __init__(self, mains, appliances,
               window_size, batch_size, shuffle=False):
    self.mains = mains
    self.appliances = appliances
    self.window_size = window_size
    self.batch_size = batch_size
    if self.window_size % 2 == 0:
      self.indices = np.arange(len(self.mains) - self.window_size)
    else:
      self.indices = np.arange(len(self.mains) - self.window_size - 1)
    self.shuffle = shuffle
  
  def __len__(self):

    return math.ceil(len(self.indices) / self.batch_size)
  
  def __getitem__(self, idx):

    mains_batch = []
    appliances_batch = []
    if idx == self.__len__() - 1:
      inds = self.indices[idx * self.batch_size:] # for data shuffling (if enabled)
    else:
      inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size] # for data shuffling (if enabled)
    for i in inds:
      main_sample = self.mains[i:i+self.window_size]
      appliance_sample = self.appliances[i+math.ceil(self.window_size/2)]
      mains_batch.append(main_sample)
      appliances_batch.append(appliance_sample)

    # Reshape is needed to make data compatible with the network input_shape.
    mains_batch_np = np.array(mains_batch)
    mains_batch_np = np.reshape(mains_batch_np, 
                                (mains_batch_np.shape[0],
                                 mains_batch_np.shape[1],
                                 1))
      
    return mains_batch_np, np.array(appliances_batch)

  def on_epoch_end(self):
    if self.shuffle:
      np.random.shuffle(self.indices)