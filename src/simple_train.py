"""
Simplified training script that demonstrates the ML pipeline without PyTorch GPU issues.
Uses scikit-learn for a quick sentiment analysis model.
"""

import os
import json
import pickle
import yaml
import mlflow
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime


def load_config(config_path='src/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data():
    """Load preprocessed train and test data"""
    train_path = 'data/processed/train.pkl'
    test_path = 'data/processed/test.pkl'
    
    print(f"Loading training data from {train_path}...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(test_data)} test samples")
    
    return train_data, test_data


def train_model(train_data, test_data, config):
    """Train a simple sentiment classification model"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    mlflow.set_experiment('sentiment-analysis')
    
    with mlflow.start_run():
        # Extract texts and labels
        X_train = train_data['text'].values
        y_train = train_data['label'].values
        X_test = test_data['text'].values
        y_test = test_data['label'].values
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        print("Vectorizing text data...")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train Random Forest classifier
        print("Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_vec)
        y_test_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        print("\n" + "-"*80)
        print("TRAINING RESULTS")
        print("-"*80)
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:  {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall:    {test_recall:.4f}")
        print(f"Test F1 Score:  {test_f1:.4f}")
        
        # Log parameters to MLflow
        mlflow.log_params({
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_features': 1000,
            'vectorizer': 'TfidfVectorizer',
            'ngram_range': '(1, 2)',
        })
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
        })
        
        # Save model artifacts
        model_path = Path('models/trained')
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        with open(model_path / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save vectorizer
        with open(model_path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Save model info
        model_info = {
            'model_type': 'RandomForest',
            'accuracy': float(test_accuracy),
            'f1_score': float(test_f1),
            'timestamp': datetime.now().isoformat(),
            'training_samples': len(train_data),
            'test_samples': len(test_data),
        }
        
        with open(model_path / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ MLflow run ID: {mlflow.active_run().info.run_id}")
        
        return model, vectorizer, {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
        }


if __name__ == '__main__':
    try:
        # Load configuration
        config = load_config()
        
        # Load data
        train_data, test_data = load_data()
        
        # Train model
        model, vectorizer, metrics = train_model(train_data, test_data, config)
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext steps:")
        print("  1. View metrics: mlflow ui")
        print("  2. Run evaluation: python src/evaluate.py")
        print("  3. Test inference: python src/inference.py")
        print("  4. Start API: python app/api.py")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
