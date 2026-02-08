"""
Download and prepare the sentiment analysis dataset.

Downloads the Microsoft Reviews dataset from HuggingFace and saves it locally.
"""

import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset


def download_dataset(dataset_name='Microsoft/reviews', output_path='data/raw/reviews.csv', num_samples=None):
    """
    Download dataset from HuggingFace and save to CSV.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_path: Path to save the CSV file
        num_samples: Limit number of samples (None = all)
    """
    print(f"Downloading {dataset_name} dataset...")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name)
        
        # Get the appropriate split
        if 'train' in dataset.keys():
            data = dataset['train']
        else:
            data = dataset['default']
        
        # Convert to pandas
        df = data.to_pandas()
        
        # Limit samples if specified
        if num_samples:
            df = df.head(num_samples)
            print(f"Limited to {num_samples} samples")
        
        # Rename columns for consistency (dataset dependent)
        if 'text' not in df.columns and 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
        
        if 'label' not in df.columns and 'rating' in df.columns:
            # Convert rating to binary sentiment (1-2 stars = 0, 4-5 stars = 1)
            df['label'] = (df['rating'] >= 4).astype(int)
            df = df.drop('rating', axis=1)
        
        # Keep only text and label columns
        if set(['text', 'label']).issubset(df.columns):
            df = df[['text', 'label']]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Creating sample dataset for demonstration...")
        
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'text': [
                'This movie was absolutely amazing! Best film I have seen in years.',
                'Terrible waste of time. Boring and predictable.',
                'It was okay, nothing special but watchable.',
                'Outstanding performance by the cast!',
                'Awful movie, worst experience ever.',
                'Great cinematography and amazing soundtrack.',
                'Disappointing and overhyped.',
                'A masterpiece of modern cinema.',
                'Not worth watching at all.',
                'Surprisingly good, exceeded expectations.'
            ] * 50,  # Repeat for more samples
            'label': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 50
        })
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(output_path, index=False)
        print(f"✓ Sample dataset created at {output_path}")
        print(f"  Total samples: {len(sample_data)}")
        
        return sample_data


if __name__ == "__main__":
    # Download with limit for quick testing
    df = download_dataset(output_path='data/raw/reviews.csv', num_samples=1000)
    
    print(f"\nDataset ready for preprocessing!")
    print(f"Next step: python src/data_loader.py")
