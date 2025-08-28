from Utils.load_models import model_classify
import numpy as np
from pathlib import Path

def _classify(dataset_name, time_series, base_path: Path = Path(".")):
    model_dir = base_path / "models" / dataset_name
    # Find the first model file in the directory, as the exact name is not known
    model_path = next(model_dir.glob("*.pth"), None) 
    if model_path is None:
        model_path = next(model_dir.glob("*.pkl"), None)

    if model_path is None:
        raise FileNotFoundError(f"No model found for dataset {dataset_name} in {model_dir}")

    print("Full path:", model_path)
    pred = model_classify(model_path=str(model_path), time_series=time_series,num_classes=2)
    return pred


    