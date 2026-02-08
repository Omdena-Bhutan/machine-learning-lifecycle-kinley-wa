"""
Simplified inference script for sentiment prediction using the trained model.
"""

import pickle
import json
from pathlib import Path


def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    model_path = Path('models/trained')
    
    with open(model_path / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_path / 'vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(model_path / 'model_info.json', 'r') as f:
        model_info = json.load(f)
    
    return model, vectorizer, model_info


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text"""
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    sentiment = 'positive' if prediction == 1 else 'negative'
    confidence = float(max(probabilities))
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence, 4),
        'probabilities': {
            'negative': round(float(probabilities[0]), 4),
            'positive': round(float(probabilities[1]), 4)
        }
    }


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS - INFERENCE")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, vectorizer, model_info = load_model_and_vectorizer()
    print(f"✓ Model loaded successfully")
    print(f"✓ Model accuracy: {model_info['accuracy']:.4f}")
    
    # Test samples
    test_samples = [
        "This movie was absolutely fantastic! I loved every moment.",
        "Terrible film. Waste of time and money. Very disappointed.",
        "It was okay, nothing special but watchable.",
        "Amazing performance by the actors! Highly recommended!",
        "Boring and predictable. Could not finish watching it.",
    ]
    
    print("\n" + "-"*80)
    print("TEST PREDICTIONS:")
    print("-"*80)
    
    for text in test_samples:
        result = predict_sentiment(text, model, vectorizer)
        
        print(f"\nText: {result['text'][:50]}...")
        print(f"  Sentiment: {result['sentiment'].upper()}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities - Negative: {result['probabilities']['negative']:.4f}, "
              f"Positive: {result['probabilities']['positive']:.4f}")
    
    print("\n" + "="*80)
    print("✓ INFERENCE COMPLETED")
    print("="*80)
    print("\nTo use the API for predictions, run:")
    print("  python app/api_simple.py")
    print("\nThen test with:")
    print('  curl -X POST http://localhost:5001/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"text": "This movie was amazing!"}\'')
    print()
