"""
Flask API for sentiment analysis predictions using the trained model.
"""

from flask import Flask, request, jsonify
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model storage
MODEL = None
VECTORIZER = None
MODEL_INFO = None


def load_model():
    """Load the trained model and vectorizer"""
    global MODEL, VECTORIZER, MODEL_INFO
    
    try:
        model_path = Path('models/trained')
        
        # Load model
        with open(model_path / 'model.pkl', 'rb') as f:
            MODEL = pickle.load(f)
        logger.info("✓ Model loaded successfully")
        
        # Load vectorizer
        with open(model_path / 'vectorizer.pkl', 'rb') as f:
            VECTORIZER = pickle.load(f)
        logger.info("✓ Vectorizer loaded successfully")
        
        # Load model info
        with open(model_path / 'model_info.json', 'r') as f:
            MODEL_INFO = json.load(f)
        logger.info("✓ Model info loaded")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': MODEL is not None
    }), 200


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    if MODEL_INFO is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model': MODEL_INFO,
        'api_version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict sentiment for a single text input.
    
    Request body:
    {
        "text": "This movie was great!"
    }
    
    Response:
    {
        "text": "This movie was great!",
        "sentiment": "positive",
        "confidence": 0.85,
        "timestamp": "2026-02-07T22:45:00.000000"
    }
    """
    try:
        if MODEL is None or VECTORIZER is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request body'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text field cannot be empty'}), 400
        
        # Vectorize and predict
        text_vec = VECTORIZER.transform([text])
        prediction = MODEL.predict(text_vec)[0]
        probabilities = MODEL.predict_proba(text_vec)[0]
        confidence = float(max(probabilities))
        
        # Map prediction to sentiment label
        sentiment = 'positive' if prediction == 1 else 'negative'
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'negative': round(float(probabilities[0]), 4),
                'positive': round(float(probabilities[1]), 4)
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple text inputs.
    
    Request body:
    {
        "texts": [
            "This movie was great!",
            "I did not enjoy this film"
        ]
    }
    
    Response:
    {
        "predictions": [
            {
                "text": "This movie was great!",
                "sentiment": "positive",
                "confidence": 0.85
            },
            ...
        ],
        "count": 2,
        "timestamp": "2026-02-07T22:45:00.000000"
    }
    """
    try:
        if MODEL is None or VECTORIZER is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field in request body'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': '"texts" must be a non-empty list'}), 400
        
        # Vectorize and predict
        texts_vec = VECTORIZER.transform(texts)
        predictions = MODEL.predict(texts_vec)
        probabilities = MODEL.predict_proba(texts_vec)
        
        # Format results
        results = []
        for i, text in enumerate(texts):
            sentiment = 'positive' if predictions[i] == 1 else 'negative'
            confidence = float(max(probabilities[i]))
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 4)
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'GET /info',
            'POST /predict',
            'POST /predict-batch'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS API")
    print("="*80)
    print("\nLoading model...")
    
    if load_model():
        print("✓ Model loaded successfully!")
        print("\n" + "-"*80)
        print("API ENDPOINTS:")
        print("-"*80)
        print("GET  http://localhost:5001/health         - Check API status")
        print("GET  http://localhost:5001/info           - Get model information")
        print("POST http://localhost:5001/predict        - Predict single text")
        print("POST http://localhost:5001/predict-batch  - Predict multiple texts")
        print("-"*80)
        print("\nExample request:")
        print('curl -X POST http://localhost:5001/predict \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"text": "This movie was amazing!"}\'')
        print("\n" + "="*80)
        print("Starting server on http://localhost:5001")
        print("="*80 + "\n")
        
        app.run(host='localhost', port=5001, debug=False, use_reloader=False)
    else:
        print("✗ Failed to load model. Exiting.")
        sys.exit(1)
