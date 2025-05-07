import pandas as pd

def read_csv_data(path_to_dataset):

  data_df = pd.read_csv(path_to_dataset)
  power_series = data_df['power']
  power_series.index = data_df['timestamp']
  return power_series