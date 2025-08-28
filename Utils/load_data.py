import numpy as np
import pandas as pd
from sklearn import preprocessing
from pathlib import Path

"""
This code has been extracted from: 
https://github.com/BrigtHaavardstun/kSimplification
"""


def read_numpy(dataset_name: str, file_name: str, base_path: Path = Path(".")) -> np.ndarray:
    """
    Parse the data from a numpy file.
    :param dataset_name: The name of the dataset directory.
    :param file_name: The name of the .npy file.
    :param base_path: The base path for session data.
    :return: A numpy array with the data.
    """
    folder = base_path / "data" / dataset_name
    file_location = folder / file_name
    array_2d = np.load(file_location)
    return array_2d


def zero_indexing_labels(current_labels: np.ndarray, dataset: str, dataset_type: str, base_path: Path = Path(".")) -> np.ndarray:
    """
    Encodes the labels as zero index.
    For instance: labels: e.g. 1,2,3,4,... -> go to -> labels: 0,1,2,3,...

    :param current_labels:
    :param dataset:
    :param base_path: The base path for session data.
    :return:
    """
    try:
        training_labels = load_dataset_org_labels(dataset, data_type="TRAIN", base_path=base_path)
        test_labels = load_dataset_org_labels(dataset, data_type="TEST", base_path=base_path)
        validation_labels = load_dataset_org_labels(dataset, data_type="VALIDATION", base_path=base_path)
        orig_labels = np.concatenate([training_labels, test_labels, validation_labels], axis=0)
    except:
        orig_labels = load_dataset_org_labels(dataset, dataset_type, base_path=base_path)
    le = preprocessing.LabelEncoder()
    le.fit(orig_labels)
    transformed_labels = le.transform(current_labels)
    return np.asarray(transformed_labels)


def load_data_set_full(dataset_name: str, data_type: str = "TRAIN", base_path: Path = Path(".")) -> np.ndarray:
    file_name = f"{dataset_name}_{data_type}.npy"
    array_2d = read_numpy(dataset_name, file_name, base_path=base_path)
    return array_2d


def load_dataset(dataset_name: str, data_type: str = "TRAIN", base_path: Path = Path(".")) -> np.ndarray:
    """
    Load all time series in {train/test} dataset.
    :param data_type:
    :param dataset_name:
    :param base_path: The base path for session data.
    :return: 2D numpy array
    """
    array_2d = load_data_set_full(dataset_name=dataset_name, data_type=data_type, base_path=base_path)
    
    # Remove the first column (index 0) along axis 1 (columns)
    data = np.delete(array_2d, 0, axis=1)
    return data


def load_dataset_org_labels(dataset_name: str, data_type: str = "TRAIN", base_path: Path = Path(".")) -> np.ndarray:
    """
    Load the labels from the dataset
    :param data_type:
    :param dataset_name:
    :param base_path: The base path for session data.
    :return:
    """
    array_2d = load_data_set_full(dataset_name, data_type=data_type, base_path=base_path)

    # Keep only the first column (index 0)
    array_2d = array_2d[:, 0]
    return array_2d


def load_dataset_labels(dataset_name, data_type: str = "TRAIN", base_path: Path = Path(".")) -> np.ndarray:
    """
    Load the labels AND onehot encode them.
    :param data_type:
    :param dataset_name:
    :param base_path: The base path for session data.
    :return:
    """
    labels_current = load_dataset_org_labels(dataset_name, data_type=data_type, base_path=base_path)
    zero_indexed = zero_indexing_labels(labels_current, dataset_name, data_type, base_path=base_path)
    return zero_indexed

def load_raw_dataset_labels(dataset_name, data_type: str = "TRAIN", base_path: Path = Path(".")) -> np.ndarray:
    """
    Load the labels AND onehot encode them.
    :param data_type:
    :param dataset_name:
    :param base_path: The base path for session data.
    :return:
    """
    labels_current = load_dataset_org_labels(dataset_name, data_type=data_type, base_path=base_path)
    return labels_current

def get_time_series(dataset_name: str, data_type:str, instance_nr: int, base_path: Path = Path(".")):
    all_time_series = load_dataset(dataset_name, data_type=data_type, base_path=base_path)
    return all_time_series[instance_nr]

def test():
    data = load_dataset("Chinatown", data_type="VALIDATION")
    print(data.shape)
    print(data)

def normalize_data(dataset_name: str, data_type: str = "TRAIN", base_path: Path = Path(".")):
    dataset = load_dataset(dataset_name, data_type, base_path=base_path)
    
    max_over_all = np.max(dataset)
    min_over_all = np.min(dataset)
    
    dataset = (dataset - min_over_all) / (max_over_all - min_over_all + 1e-8)
    file_path_name = base_path / "data" / dataset_name / f"{dataset_name}_{data_type}_normalized.npy"
    labels = load_raw_dataset_labels(dataset_name, data_type, base_path=base_path)
    dataset = np.hstack((labels.reshape(-1, 1), dataset))
    assert not np.isnan(dataset).any(), f"NaN values in the dataset {dataset}."
    np.save(file_path_name, dataset)

if __name__ == "__main__":
    test()