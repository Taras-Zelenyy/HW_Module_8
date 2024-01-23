import unittest
import os
import sys
import json

import pandas as pd

# Setting the path to the project root directory and configuration file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = 'settings.json'

from data_process.data_generation import IrisDataset  # Import your IrisDataset class
from utils import get_project_dir


class TestDataGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading the configuration
        with open(CONF_FILE, "r", encoding="utf-8") as file:
            conf = json.load(file)
        
        cls.data_dir = get_project_dir(conf['general']['data_dir'])

        # Path to training and inference data
        cls.train_path = os.path.join(cls.data_dir, 'temporary_training_data.csv')
        cls.inference_path = os.path.join(cls.data_dir, 'temporary_infere_data.csv')

        cls.created_files = []
        cls.created_files.extend([cls.train_path, cls.inference_path])
            
    def test_dataset_creation(self):
        """Testing the creation and loading of a dataset."""
        iris_dataset = IrisDataset()
        iris_dataset.create(self.train_path, self.inference_path)

        # Check the availability of the dataset
        self.assertIsNotNone(iris_dataset.df, "Dataset should be created.")
        self.assertIsNotNone(iris_dataset.train_df, "Train dataset should be created.")
        self.assertIsNotNone(iris_dataset.inference_df, "Inference dataset should be created.")

        # Checking the data type
        self.assertIsInstance(iris_dataset.train_df, pd.DataFrame, "Train dataset should be a pandas DataFrame.")
        self.assertIsInstance(iris_dataset.inference_df, pd.DataFrame, "Inference dataset should be a pandas DataFrame.")

        # Check the data form
        self.assertGreater(iris_dataset.train_df.shape[0], 0, "Train dataset should have more than 0 rows.")
        self.assertGreater(iris_dataset.inference_df.shape[0], 0, "Inference dataset should have more than 0 rows.")
        
        # Check the number of rows
        self.assertEqual(iris_dataset.train_df.shape[0], 120, "Train dataset should have 120 rows.")
        self.assertEqual(iris_dataset.inference_df.shape[0], 30, "Inference dataset should have 30 rows.")

        # Checking if datasets contain the expected columns
        expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
        self.assertTrue(all(column in iris_dataset.train_df.columns for column in expected_columns), "Train dataset should have all expected columns.")
        self.assertTrue(all(column in iris_dataset.inference_df.columns for column in expected_columns), "Inference dataset should have all expected columns.")
    
        
    @classmethod
    def tearDownClass(cls):
        # Clean up: remove temporary files created for testing
        for file_path in cls.created_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                
                
if __name__ == '__main__':
    unittest.main()