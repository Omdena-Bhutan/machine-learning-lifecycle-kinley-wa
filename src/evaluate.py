import json
import pickle
import yaml
import mlflow
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_model(model_path='models/trained/model.pkl'):
    """Load trained model from pickle file"""
    print(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None


def load_test_data(test_path='data/processed/test.pkl'):
    """Load test data from pickle file"""
    print(f"Loading test data from {test_path}")
    try:
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        return test_data
    except FileNotFoundError:
        print(f"Test data file not found at {test_path}")
        return None


def generate_predictions(model, test_data, model_name="distilbert-base-uncased"):
    """Generate predictions on test data"""
    print("Generating predictions...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    inputs = tokenizer(
        test_data['text'].tolist(),
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=1)
    
    return predictions


def evaluate_model(model, test_data, model_name="distilbert-base-uncased", threshold=0.5):
    """
    Evaluate model on test data and calculate metrics.
    """
    print("Evaluating model...")
    
    predictions = generate_predictions(model, test_data, model_name)
    true_labels = test_data['label'].values
    
    metrics = {
        'accuracy': float(accuracy_score(true_labels, predictions)),
        'precision': float(precision_score(true_labels, predictions, average='weighted', zero_division=0)),
        'recall': float(recall_score(true_labels, predictions, average='weighted', zero_division=0)),
        'f1': float(f1_score(true_labels, predictions, average='weighted', zero_division=0)),
    }
    
    print(f"Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics


def create_confusion_matrix(true_labels, predictions, output_path='eval_metrics.json'):
    """Create and save confusion matrix visualization"""
    print("Creating confusion matrix...")
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    Path('results').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/eval_metrics_cm.png')
    print("Confusion matrix saved to results/eval_metrics_cm.png")


def save_metrics(metrics, output_path='results/eval_metrics.json'):
    """Save evaluation metrics to JSON file"""
    print(f"Saving metrics to {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    metrics_json = {k: float(v) if isinstance(v, (np.ndarray, np.floating)) else v 
                    for k, v in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to {output_path}")


def log_metrics_to_mlflow(metrics):
    """Log evaluation metrics to MLflow"""
    print("Logging metrics to MLflow...")
    
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    
    if Path('results/eval_metrics_cm.png').exists():
        mlflow.log_artifact('results/eval_metrics_cm.png')
    
    if Path('results/eval_metrics.json').exists():
        mlflow.log_artifact('results/eval_metrics.json')
    
    print("Metrics logged to MLflow")


def error_analysis(true_labels, predictions, test_data):
    """Perform error analysis on predictions"""
    print("Performing error analysis...")
    
    error_mask = true_labels != predictions
    errors = test_data[error_mask]
    
    print(f"\nTotal errors: {len(errors)}")
    print(f"Error rate: {len(errors) / len(test_data) * 100:.2f}%")
    
    if len(errors) > 0:
        print("\nSample misclassifications:")
        print(errors.head())
    
    return errors


if __name__ == "__main__":
    # Load configuration
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Load model and test data
    model = load_model()
    test_data = load_test_data()
    
    # Evaluate model
    if model is not None and test_data is not None:
        metrics = evaluate_model(
            model,
            test_data,
            model_name=params['train']['model_name'],
            threshold=params['evaluate'].get('threshold', 0.5)
        )
        
        # Save metrics
        save_metrics(metrics)
        
        # Log to MLflow
        with mlflow.start_run():
            log_metrics_to_mlflow(metrics)
        
        print("\nEvaluation completed!")
    else:
        print("Error: Model or test data not loaded properly")
