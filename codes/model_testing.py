import data_loading
import data_preprocessing
import data_ingestion
import numpy as np
import tensorflow as tf
import metrics
import math

def test_model(appliance_name, main_path, appliance_path, model_path, 
               window_size, batch_size, rescaling=None, 
               appliance_min_power=0.0, appliance_max_power=1.0,
               appliance_mean_power=0.0, appliance_std_power=1.0,
               main_min_power=0.0, main_max_power=1.0,
               main_mean_power=0.0, main_std_power=1.0):
 
  # Data Loading
  print('Data Loading...', end='')
  appliance_test = data_loading.read_csv_data(appliance_path).values
  main_test = data_loading.read_csv_data(main_path).values
  print('Done.')
  # Save ground truth to eventually return it
  ground_truth = appliance_test + np.zeros(appliance_test.shape)
  # Zero-padding
  print('Zero padding...', end='')
  appliance_test = data_preprocessing.zero_pad(appliance_test, window_size)
  main_test = data_preprocessing.zero_pad(main_test, window_size)
  print('Done.')
  # Rescaling (if required)
  if rescaling is not None:
    print('Rescaling...', end='')
    if rescaling == 'standardize':
      appliance_test = data_preprocessing.standardize_data(appliance_test,
                                                           appliance_mean_power,
                                                           appliance_std_power)
      main_test = data_preprocessing.standardize_data(main_test,
                                                      main_mean_power,
                                                      main_std_power)
    if rescaling == 'normalize':
      appliance_test = data_preprocessing.normalize_data(appliance_test, 
                                                         appliance_min_power, 
                                                         appliance_max_power)
      main_test = data_preprocessing.normalize_data(main_test,
                                                    main_min_power,
                                                    main_max_power)
     
    print('Done.')
  # Preparing data ingestion
  print('Preparing data ingestion...', end='')
  test_ingestor = data_ingestion.DataIngestor(main_test, appliance_test, 
                                              window_size, batch_size)
  print('Done.')
  # Load model from the given file path
  model = tf.keras.models.load_model(model_path)
  test_steps = test_ingestor.__len__()
  print('Predicting...', end='')
  predicted_values = model.predict(x=test_ingestor, steps=test_steps)
  if rescaling == 'normalize':
    predicted_values *= (appliance_max_power - appliance_min_power)
    predicted_values += appliance_min_power
  if rescaling == 'standardize':
    predicted_values *= appliance_std_power
    predicted_values += appliance_mean_power
    predicted_values[predicted_values < 0.0] = 0.0
  print('Done.')
  return ground_truth, predicted_values
  