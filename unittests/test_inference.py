import unittest
import os
import sys
import json
import shutil

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Setting the path to the project root directory and configuration file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = 'settings.json'

# Import functions from inference.run
from inference.run import get_model_by_path, get_inference_data, predict_results, store_results, get_latest_model_path
from utils import get_project_dir


class TestRunScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading the configuration
        with open(CONF_FILE, "r", encoding="utf-8") as file:
            conf = json.load(file)
        
        cls.data_dir = get_project_dir(conf['general']['data_dir'])
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)

        cls.created_files = []

        # Path to temporary inference data
        cls.inference_data_path = os.path.join(cls.data_dir, "temp_inference_data.csv")
        cls.created_files.append(cls.inference_data_path)

        # Create temporary inference data for testing if it doesn't exist
        if not os.path.exists(cls.inference_data_path):
            cls.create_temporary_test_data(cls.inference_data_path, conf)
        
    
    @staticmethod
    def create_temporary_test_data(inference_path, conf):
        """Create temporary data for testing"""
        data = load_iris()
        df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                          columns=data['feature_names'] + ['target'])

        # Using only 10% of the data for test dataset
        _, inference_df = train_test_split(df, test_size=0.1, random_state=conf['general']['random_state'])
        inference_df.drop(columns='target', inplace=True)  # Drop the target column for inference

        # Save temporary inference data
        inference_df.to_csv(inference_path, index=False)
    
        
    def test_model_loading(self):
        """Test the ability to load the latest model."""
        model_path = get_latest_model_path()
        self.assertIsNotNone(model_path, "Latest model path should be found.")
        
        model = get_model_by_path(model_path)
        self.assertIsNotNone(model, "Model should be loaded successfully.")
    
        
    def test_inference(self):
        """Test the inference process with the latest model."""
        model_path = get_latest_model_path()
        self.assertIsNotNone(model_path, "Latest model path should be found.")
        
        model = get_model_by_path(model_path)
        inference_data = get_inference_data(self.inference_data_path)

        try:
            results = predict_results(model, inference_data)
            self.assertIsNotNone(results, "Inference should be executed without errors.")
        except Exception as e:
            self.fail(f"Inference failed with error: {e}")
    
            
    def test_result_storage(self):
        """Test the storage of inference results."""
        
        # Define a temporary directory for testing
        test_results_dir = os.path.join(ROOT_DIR, "test_results")
        if not os.path.exists(test_results_dir):
            os.makedirs(test_results_dir)

        # Create a sample results DataFrame (This is imaginary data created for the purpose of testing)
        sample_results = pd.DataFrame({'sepal length (cm)': [5.1, 5.2, 5.0],
                                       'sepal width (cm)': [3.5, 3.6, 0.2],
                                       'petal length (cm)': [1.4, 1.3, 1.1],
                                       'petal width (cm)': [0.2, 0.2, 1.0],
                                       'predicted': [0, 1, 2]})

        # Store the sample results
        results_path = os.path.join(test_results_dir, "test_results.csv")
        store_results(sample_results, results_path)

        self.assertTrue(os.path.exists(results_path), "Results should be saved to the specified location.")

        # Clean up the temporary directory
        os.remove(results_path)
        os.rmdir(test_results_dir)
    
        
    @classmethod
    def tearDownClass(cls):
        # Clean up: remove temporary files created for testing
        for file_path in cls.created_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove created directories if empty
        if os.path.exists(cls.data_dir) and not os.listdir(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        
        
if __name__ == '__main__':
    unittest.main()