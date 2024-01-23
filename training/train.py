"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassF1Score,
                                         MulticlassPrecision,
                                         MulticlassRecall)


# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_project_dir, configure_logging

CONF_FILE = "settings.json"

# Loads configuration settings from JSON
try:
    with open(CONF_FILE, "r", encoding="utf-8") as file:
        conf = json.load(file)
except FileNotFoundError:
    print(f"Error: Configuration file {CONF_FILE} not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Configuration file {CONF_FILE} is not a valid JSON.")
    sys.exit(1)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")


class DataProcessor:
    """ Handles loading data, spliting data and converting to tensor"""
    def __init__(self) -> None:
        pass


    def setup(self, data_path: str) -> tuple:
        """Set up the data module and prepare train and test datasets."""
        logging.info("Loading and preparing data...")

        # Downloading data
        data = self.data_extraction(data_path)
        feature_names = [
            'sepal length (cm)', 
            'sepal width (cm)',
            'petal length (cm)', 
            'petal width (cm)'
        ]

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(data, feature_names)

        # Log the shape of training and test data
        logging.info(f"Shape of training data: {X_train.shape}, Training labels: {y_train.shape}")
        logging.info(f"Shape of test data: {X_test.shape}, Test labels: {y_test.shape}")
        
        # Convert to tensors
        X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors = (
            self.convert_to_tensors(X_train, y_train, X_test, y_test)
        )

        self.train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
        self.test_dataset = TensorDataset(X_test_tensors, y_test_tensors)
        
        return self.train_dataset, self.test_dataset


    def data_extraction(self, path: str) -> pd.DataFrame:
        """Load and return a DataFrame from the specified CSV file."""
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    
    def split_data(
        self, dataset: pd.DataFrame, feature_names: list, test_size: float = 0.25
    ) -> tuple:
        """
        Split the dataset into training and test sets based
        on the provided feature names and test size.
        """
        logging.info("Splitting data into training and test sets...")
        X = dataset[feature_names]
        y = dataset['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=conf['general']['random_state'], stratify=y
        )

        return X_train, X_test, y_train, y_test

    
    def convert_to_tensors(
        self, X_train, y_train, X_test, y_test
    ):
        """Convert training and test datasets from Pandas DataFrames/Series to PyTorch tensors."""
        logging.info("Converting data to tensors...")  
        X_train_tensors = torch.tensor(X_train.values, dtype=torch.float32) 
        y_train_tensors = torch.tensor(y_train.values, dtype=torch.long)
        X_test_tensors = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensors = torch.tensor(y_test.values, dtype=torch.long)

        return X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors


class IrisClassifier(nn.Module):
    """Base line NN model"""
    
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(IrisClassifier, self).__init__()
        self.input_layer = nn.Linear(in_features=input_dim, out_features=128)
        self.hidden_layer1 = nn.Linear(in_features=128, out_features=64)
        self.output_layer = nn.Linear(in_features=64, out_features=output_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.output_layer(x)
        return x


class Training:
    """Handles model training, evaluation, and saving."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize the training class with a given model."""
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def run_training(self, train_loader: DataLoader, test_loader: DataLoader, 
                     num_epochs: int, output_dim: int, out_path: str = None) -> None:
        """Run the model training and evaluation process."""
        logging.info("Running training...")

        # Logging of hyperparameters
        mlflow.log_params({
            'num_epochs': num_epochs,
            'optimizer': type(self.optimizer).__name__,
            'loss_function': type(self.criterion).__name__,
        })
        
        start_time = time.time()
        self.train(train_loader, num_epochs)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")

        self.evaluate(test_loader, output_dim)  
        self.save(out_path)


    def train(self, train_loader: DataLoader, num_epochs: int) -> None:
        """Train the model using the provided data loader and number of epochs."""
        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    def evaluate(self, test_loader: DataLoader, output_dim: int) -> None:
        """Evaluate the model on the test dataset and log metrics."""
        accuracy_metric = MulticlassAccuracy(num_classes=output_dim)
        precision_metric = MulticlassPrecision(num_classes=output_dim, average='macro')
        recall_metric = MulticlassRecall(num_classes=output_dim, average='macro')
        f1_metric = MulticlassF1Score(num_classes=output_dim, average='macro')

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                accuracy_metric(preds, labels)
                precision_metric(preds, labels)
                recall_metric(preds, labels)
                f1_metric(preds, labels)
        
        mlflow.log_metrics({
            'Accuracy': accuracy_metric.compute().item(),
            'Precision': precision_metric.compute().item(),
            'Recall': recall_metric.compute().item(),
            'F1 Score': f1_metric.compute().item()
        })
        
        logging.info(f'\nTest data scores:\n'
                     f'Accuracy: {accuracy_metric.compute()},\n'
                     f'Precision: {precision_metric.compute()},\n'
                     f'Recall: {recall_metric.compute()},\n'
                     f'F1: {f1_metric.compute()}\n')
        

    def save(self, path: str) -> None:
        """Save the model to the given path."""
        try:
            logging.info("Saving the model...")
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            if not path:
                path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
            else:
                path = os.path.join(MODEL_DIR, path)

            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logging.info(f"Model saved: {path}")
        except Exception as e:
            logging.error(f"Error during model saving: {e}")
            sys.exit(1)
    

def main():
    configure_logging()
    mlflow.pytorch.autolog()  
    mlflow.set_experiment('Iris_Classification')
    
    with mlflow.start_run():
        configure_logging()  
        logging.info("Starting the training process...")
        
        try:
            data_proc = DataProcessor()
            train_dataset, test_dataset = data_proc.setup(TRAIN_PATH)
        except Exception as e:
            logging.error(f"Error during data processing: {e}")
            sys.exit(1)
        
        try:
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=conf['train']['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=conf['train']['batch_size'], shuffle=False)     
        except Exception as e:
            logging.error(f"Error during data loaders creating: {e}")
            sys.exit(1)

        try:
            # Initialize the model
            model = IrisClassifier(input_dim=4, output_dim=3)
        except Exception as e:
            logging.error(f"Error during model initialization: {e}")
            sys.exit(1)
        
        try:
            tr = Training(model)
            tr.run_training(train_loader, test_loader, num_epochs=conf['train']['max_epochs'], output_dim=3)
            logging.info("Training process completed.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            sys.exit(1)
        
    mlflow.end_run()

if __name__ == "__main__":
    main()