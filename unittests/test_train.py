import unittest
import os
import sys
import pandas as pd
import numpy as np
import torch
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import shutil

# Setting the path to the project root directory and configuration file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = 'settings.json'

from training.train import IrisClassifier, DataProcessor, Training
from utils import get_project_dir, configure_logging

class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading the configuration
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        
        cls.data_dir = get_project_dir(conf['general']['data_dir'])
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)

        cls.created_files = []

        # Path to temporary training data
        cls.train_path = os.path.join(cls.data_dir, 'temporary_training_data.csv')
        cls.created_files.append(cls.train_path)

        # Create temporary training data for testing if it doesn't exist
        if not os.path.exists(cls.train_path):
            cls.create_temporary_train_data(cls.train_path, conf)
    
    
    @staticmethod
    def create_temporary_train_data(train_path, conf):
        data = load_iris()
        df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                          columns=data['feature_names'] + ['target'])

        # Using only 10% of the data for test dataset
        _, train_df = train_test_split(df, test_size=0.1, random_state=conf['general']['random_state'])

        # Save temporary training data
        train_df.to_csv(train_path, index=False)


    def test_model_initialization(self):
        """Test whether the model initializes without errors."""
        model = IrisClassifier(input_dim=4, output_dim=3)
        self.assertIsNotNone(model)


    def test_default_training(self):
        """Test whether the model can be trained with default parameters."""
        data_processor = DataProcessor()
        train_dataset, test_dataset = data_processor.setup(self.train_path)

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize the model
        model = IrisClassifier(input_dim=4, output_dim=3)

        training = Training(model)
        training.run_training(train_loader, test_loader, num_epochs=10, output_dim=3)

        # Check if model parameters have been updated after training
        self.assertTrue(any(param.requires_grad for param in model.parameters()))


    @classmethod
    def tearDownClass(cls):
        # Clean up: remove temporary files created for testing
        for file_path in cls.created_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
if __name__ == '__main__':
    unittest.main()
