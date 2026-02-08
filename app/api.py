import sys
import os
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
from src.inference import predict, predict_batch
from src.model import SentimentModel
import logging
from datetime import datetime

# Get the app directory for serving static files
APP_DIR = Path(__file__).parent

app = Flask(__name__, static_folder=str(APP_DIR), static_url_path='')
CORS(app)  # Enable CORS for API calls from frontend
logger = logging.getLogger(__name__)

# TODO: Initialize model at startup
MODEL = None
MODEL_VERSION = "1.0.0"
DEPLOYMENT_TIME = datetime.now().isoformat()


@app.before_request
def load_model():
    """TODO: Load model on first request"""
    global MODEL
    if MODEL is None:
        try:
            from src.inference import SentimentPredictor
            MODEL = SentimentPredictor()
            MODEL.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


@app.route('/', methods=['GET'])
def home():
    """Serve the dashboard HTML file"""
    try:
        return send_from_directory(str(APP_DIR), 'index.html')
    except FileNotFoundError:
        return jsonify({
            'status': 'success',
            'message': 'Sentiment Analysis API',
            'version': MODEL_VERSION,
            'note': 'Open http://localhost:5000 in your browser to use the dashboard',
            'endpoints': {
                'POST /predict': 'Single sentiment prediction',
                'POST /predict-batch': 'Batch sentiment predictions',
                'GET /health': 'Health check',
                'GET /metrics': 'Model metrics',
                'GET /version': 'API version info'
            }
        }), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    TODO: Predict sentiment for input text.
    
    Request body:
    {
        "text": "Movie review text here"
    }
    
    Response:
    {
        "sentiment": "positive|negative",
        "confidence": 0.95,
        "text": "input text"
    }
    """
    try:
        # TODO: Validate request has JSON content
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        
        # TODO: Validate input contains 'text' field
        if 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'status': 'error'
            }), 400
        
        text = data['text']
        
        # TODO: Validate text is not empty
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text cannot be empty',
                'status': 'error'
            }), 400
        
        try:
            result = MODEL.predict_single(text)
            return jsonify({
                'status': 'success',
                'data': result,
                'timestamp': datetime.now().isoformat()
            }), 200
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch_endpoint():
    """
    TODO: Predict sentiment for multiple texts.
    
    Request body:
    {
        "texts": ["text1", "text2", ...],
        "batch_size": 32
    }
    
    Response:
    {
        "status": "success",
        "predictions": [
            {"text": "...", "sentiment": "...", "confidence": ...},
            ...
        ]
    }
    """
    try:
        # TODO: Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # TODO: Validate texts field
        if 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        texts = data['texts']
        batch_size = data.get('batch_size', 32)
        
        # TODO: Validate texts are not empty
        if len(texts) == 0:
            return jsonify({'error': 'texts list cannot be empty'}), 400
        
        try:
            predictions = list(MODEL.predict_batch(texts, batch_size=batch_size))
            return jsonify({
                'status': 'success',
                'predictions': predictions,
                'count': len(predictions)
            }), 200
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"Error in batch predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    TODO: Health check endpoint for monitoring.
    
    Returns:
    {
        "status": "healthy|unhealthy",
        "model_version": "1.0.0",
        "deployed_at": "ISO timestamp"
    }
    """
    try:
        model_loaded = MODEL is not None
        
        health_status = {
            'status': 'healthy' if model_loaded else 'unhealthy',
            'model_version': MODEL_VERSION,
            'deployed_at': DEPLOYMENT_TIME,
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(health_status), 200 if model_loaded else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    TODO: Return model and API metrics.
    
    Returns:
    {
        "total_predictions": 1000,
        "avg_latency_ms": 45.2,
        "success_rate": 99.5,
        "model_version": "1.0.0"
    }
    """
    try:
        metrics_data = {
            'total_predictions': 0,
            'avg_latency_ms': 0.0,
            'success_rate': 100.0,
            'model_version': MODEL_VERSION,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(metrics_data), 200
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/version', methods=['GET'])
def version():
    """Return API and model version information"""
    return jsonify({
        'api_version': '1.0.0',
        'model_version': MODEL_VERSION,
        'deployed_at': DEPLOYMENT_TIME
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"Starting API server on http://0.0.0.0:5000")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Deployed at: {DEPLOYMENT_TIME}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
