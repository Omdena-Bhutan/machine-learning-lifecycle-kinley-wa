"""
Lightweight evaluation script using the scikit-learn trained model.
Loads `data/processed/test.pkl`, the model at `models/trained/model.pkl`, and
vectorizer at `models/trained/vectorizer.pkl`, computes metrics and prints them.
Also logs metrics to MLflow.
"""

import pickle
import mlflow
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_test_data():
    import pickle
    with open('data/processed/test.pkl','rb') as f:
        return pickle.load(f)


def load_model():
    model_path = Path('models/trained')
    with open(model_path / 'model.pkl','rb') as f:
        model = pickle.load(f)
    with open(model_path / 'vectorizer.pkl','rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


if __name__ == '__main__':
    print('\nLoading test data and model...')
    test_df = load_test_data()
    model, vectorizer = load_model()

    X_test = test_df['text'].values
    y_test = test_df['label'].values

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print('\nEvaluation Results:')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall   : {rec:.4f}')
    print(f'  F1 Score : {f1:.4f}')

    # Log to MLflow
    mlflow.set_experiment('sentiment-analysis')
    with mlflow.start_run(nested=True):
        mlflow.log_metrics({
            'eval_accuracy': acc,
            'eval_precision': prec,
            'eval_recall': rec,
            'eval_f1': f1
        })
    print('\nMetrics logged to MLflow under experiment "sentiment-analysis"')
