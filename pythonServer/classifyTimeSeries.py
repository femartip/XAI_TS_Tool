from Utils.load_models import model_classify
from Utils.load_data import read_numpy
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

    # Infer number of classes: prefer checkpoint metadata, fallback to labels in data
    num_classes = 2
    try:
        if str(model_path).endswith('.pth'):
            import torch  # local import to avoid hard dependency elsewhere
            state = torch.load(str(model_path), map_location=torch.device('cpu'))
            if isinstance(state, dict) and 'fc.weight' in state:
                out_dim = state['fc.weight'].shape[0]
                num_classes = int(out_dim) if out_dim > 1 else 2
    except Exception as e:
        print(f"Warning: could not infer num_classes from checkpoint: {e}.")

    if num_classes == 2:  # still unknown, try from data
        try:
            data_dir = base_path / "data" / dataset_name
            any_npy = next(data_dir.glob("*.npy"))
            array_2d = np.load(any_npy)
            labels = array_2d[:, 0]
            num_classes = int(len(np.unique(labels))) if len(labels) > 0 else 2
        except Exception as e:
            print(f"Warning: could not infer num_classes for {dataset_name} from data: {e}. Defaulting to 2.")

    pred = model_classify(model_path=str(model_path), time_series=time_series, num_classes=num_classes)
    return pred


    
