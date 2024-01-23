# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
# CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = "settings.json"

# Load configuration settings from JSON
try:
    logger.info("Loading configuration settings from JSON...")
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)
except FileNotFoundError:
    logger.error(f"Configuration file {CONF_FILE} not found.")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"Configuration file {CONF_FILE} is not a valid JSON.")
    sys.exit(1)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for creating Iris data set
@singleton
class IrisDataset():
    def __init__(self):
        self.df = None
        self.train_df = None
        self.inference_df = None

    def create(self, train_path: os.path, inference_path: os.path):
        """Method to create the Iris dataset."""
        try:
            logger.info("Creating Iris dataset...")
            data = load_iris()
            self.df = pd.DataFrame(
                data=np.c_[data['data'], data['target']], 
                columns=data['feature_names'] + ['target']
            )

            # Split the data into train and test sets
            self.train_df, self.inference_df = train_test_split(
                self.df, 
                test_size=conf['train']['test_size'], 
                random_state=conf['general']['random_state'], 
                stratify=self.df['target'],
            )

            logger.info("Iris dataset created.")
            
            # Log the size of used dataset before training and inference
            logger.info(f"Size of training dataset: {self.train_df.shape} records")
            logger.info(f"Size of inference dataset: {self.inference_df.shape} records")
            
            # Save the datasets
            self.save(self.train_df, train_path)
            self.save(self.inference_df, inference_path)
        except Exception as e:
            logger.error(f"Error creating the Iris dataset: {e}")
            sys.exit(1)

    def save(self, df: pd.DataFrame, out_path: os.path):
        """Method to save data."""
        try:
            logger.info(f"Saving data to {out_path}...")
            df.to_csv(out_path, index=False)
        except Exception as e:
            logger.error(f"Error saving data to {out_path}: {e}")
            sys.exit(1)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    
    try:
        # Instance of the IrisDataset
        gen = IrisDataset()
        
        # Create the datasets
        gen.create(train_path=TRAIN_PATH, inference_path=INFERENCE_PATH)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
        
    logger.info("Script completed successfully.")