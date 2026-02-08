import os
import pickle
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path


class DataPreprocessor:
    """
    Handles data loading, preprocessing, and train/test splitting.
    """
    
    def __init__(self, data_path, max_length=128, config_path="src/config.yaml"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path: Path to the raw CSV file
            max_length: Maximum sequence length for tokenization
            config_path: Path to config.yaml for parameters
        """
        self.data_path = data_path
        self.max_length = max_length
        self.config = self._load_config(config_path)
        self.df = None
        self.train_data = None
        self.test_data = None
    
    def _load_config(self, config_path):
        """Load YAML configuration file and return as dictionary"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config not found at {config_path}, using defaults")
            return {}
    
    def load_data(self):
        """Load data from CSV file using pandas"""
        print(f"Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(self.df.head())
        
        return self.df
    
    def preprocess(self):
        """Clean and preprocess text data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Preprocessing data...")
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Convert text to lowercase
        self.df['text'] = self.df['text'].str.lower()
        
        # Remove special characters and extra whitespace
        import re
        self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        self.df['text'] = self.df['text'].str.strip()
        
        print(f"Data after preprocessing: {self.df.shape}")
        return self.df
    
    def create_splits(self):
        """Create train/test splits"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() and preprocess() first.")
        
        print("Creating train/test splits...")
        
        # Extract parameters from config
        test_size = self.config.get('data', {}).get('test_size', 0.2)
        random_state = self.config.get('data', {}).get('random_state', 42)
        
        # Split data into train and test sets
        self.train_data, self.test_data = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['label']
        )
        
        print(f"Train set size: {len(self.train_data)}")
        print(f"Test set size: {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def save_splits(self, output_dir="data/processed"):
        """Save processed train/test splits to pickle files"""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Splits not created. Call create_splits() first.")
        
        print(f"Saving splits to {output_dir}...")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save train and test splits as pickle files
        train_path = os.path.join(output_dir, 'train.pkl')
        test_path = os.path.join(output_dir, 'test.pkl')
        
        with open(train_path, 'wb') as f:
            pickle.dump(self.train_data, f)
        with open(test_path, 'wb') as f:
            pickle.dump(self.test_data, f)
        
        print(f"Saved train.pkl and test.pkl to {output_dir}")
    
    def run_pipeline(self):
        """Execute the complete data preprocessing pipeline"""
        self.load_data()
        self.preprocess()
        self.create_splits()
        self.save_splits()
        print("Data preprocessing pipeline completed!")


if __name__ == "__main__":
    # Load parameters from params.yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(
        data_path=params['data']['dataset_path'],
        max_length=params['train']['max_length']
    )
    
    preprocessor.run_pipeline()
