
import numpy as np
from Utils.load_data import load_dataset

def get_time_series(dataset_name, data_type, index, base_path="./"):
    x = load_dataset(dataset_name,data_type=data_type, base_path=base_path)
    return x[index][1:]

