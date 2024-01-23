"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch


# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

try:
    # Loads configuration settings from JSON
    with open(CONF_FILE, "r", encoding="utf-8") as file:
        conf = json.load(file)
except FileNotFoundError:
    print(f"Error: Configuration file {CONF_FILE} not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Configuration file {CONF_FILE} is not a valid JSON.")
    sys.exit(1)

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r", encoding="utf-8") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Get the path of the latest saved model."""
    latest = None
    for _, _, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    if latest is None:
        logging.error(f"No model found in {MODEL_DIR}.")
        sys.exit(1)
    return os.path.join(MODEL_DIR, latest)


class CustomUnpickler(pickle.Unpickler):
    """Custom Unpickler class to handle the unpickling of specific classes."""
    def find_class(self, module, name):
        if name == 'IrisClassifier':
            from training.train import IrisClassifier
            return IrisClassifier
        return super().find_class(module, name)


def get_model_by_path(path: str):
    """Load and return the specified model."""
    try:
        with open(path, 'rb') as f:
            model = CustomUnpickler(f).load()
            logging.info(f'Path of the model: {path}')
            return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """Load and return data for inference from the specified csv file."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict results using the model and the inference data."""
    x_inference = torch.tensor(
        infer_data[
            [
                'sepal length (cm)', 
                'sepal width (cm)',
                'petal length (cm)', 
                'petal width (cm)'
            ]
        ].values,
        dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        predicted = np.argmax(model(x_inference).numpy(), axis=1)    
    infer_data['predicted'] = predicted
    return infer_data



def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in the 'results' directory."""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    try:
        pd.DataFrame(results).to_csv(path, index=False)
        logging.info(f'Results saved to {path}')
    except Exception as e:
        logging.error(f"An error occurred while saving results: {e}")
        sys.exit(1)


def main():
    """Main function."""
    configure_logging()
    args = parser.parse_args()

    try:
        model = get_model_by_path(get_latest_model_path())
    except Exception as e:
        logging.error(f"Error in model loading or processing: {e}")
        sys.exit(1)

    try:
        infer_file = args.infer_file
        infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    except Exception as e:
        logging.error(f"Error in loading inference data: {e}")
        sys.exit(1)

    try:
        start_time = time.time()
        results = predict_results(model, infer_data)
        end_time = time.time()
        logging.info(f"Inference completed in {end_time - start_time} seconds.")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        sys.exit(1)

    try:
        store_results(results, args.out_path)
    except Exception as e:
        logging.error(f"Error during results storage: {e}")
        sys.exit(1)

    logging.info(f'\nPrediction results:\n {results}')


if __name__ == "__main__":
    main()
