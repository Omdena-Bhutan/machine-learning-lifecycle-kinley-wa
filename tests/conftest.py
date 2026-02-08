"""
Pytest configuration and shared fixtures for the sentiment analysis project.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'text': [
            'This is a great movie',
            'This is a terrible film',
            'It was an okay movie',
            'Amazing performance by the actors',
            'Worst movie I have ever seen'
        ],
        'label': [1, 0, 1, 1, 0]
    })


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(sample_dataframe, temp_data_dir):
    """Create a sample CSV file for testing"""
    csv_path = temp_data_dir / 'sample_data.csv'
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_texts():
    """Sample text for inference testing"""
    return [
        'This movie was amazing! I loved it!',
        'Terrible waste of time.',
        'It was okay.',
        'Outstanding film with great actors.',
        'Worst movie ever made.'
    ]


@pytest.fixture
def mock_config():
    """Mock configuration dictionary"""
    return {
        'data': {
            'dataset_path': 'data/raw/reviews.csv',
            'test_size': 0.2,
            'random_state': 42
        },
        'train': {
            'model_name': 'distilbert-base-uncased',
            'batch_size': 32,
            'epochs': 3,
            'learning_rate': 2e-5,
            'max_length': 128
        }
    }


@pytest.fixture(scope='session')
def test_session_config():
    """Session-wide configuration"""
    return {
        'test_timeout': 300,
        'test_batch_size': 8,
        'use_gpu': False
    }
