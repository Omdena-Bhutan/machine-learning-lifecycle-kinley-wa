import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentPredictor:
    """
    Performs inference using trained sentiment analysis model.
    """
    
    def __init__(self, model_path='models/trained/model.pkl', 
                 model_name='distilbert-base-uncased'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model
            model_name: HuggingFace model identifier for tokenizer
        """
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load model and tokenizer from disk"""
        print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model from pretrained (uses cached version if available)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def predict_single(self, text, threshold=0.5):
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Predicting sentiment for: {text[:50]}...")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predicted label and confidence
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_label].item()
        
        # Map label to sentiment (0=negative, 1=positive)
        sentiment = "positive" if predicted_label == 1 else "negative"
        
        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'label': predicted_label
        }
        
        return result
    
    def predict_batch(self, texts, batch_size=32, threshold=0.5):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            threshold: Decision threshold
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Predicting sentiment for {len(texts)} texts...")
        
        predictions = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Process each prediction in batch
            for j, text in enumerate(batch):
                predicted_label = torch.argmax(probabilities[j]).item()
                confidence = probabilities[j, predicted_label].item()
                sentiment = "positive" if predicted_label == 1 else "negative"
                
                predictions.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'label': predicted_label
                })
        
        return predictions


# Convenience functions for quick predictions
def predict(text, model_path='models/trained/model.pkl', 
            model_name='distilbert-base-uncased', threshold=0.5):
    """
    Quick prediction function for single text.
    
    Args:
        text: Input text string
        model_path: Path to saved model
        model_name: HuggingFace model identifier
        threshold: Decision threshold
        
    Returns:
        Dictionary with sentiment prediction
    """
    predictor = SentimentPredictor(model_path, model_name)
    predictor.load_model()
    return predictor.predict_single(text, threshold)


def predict_batch(texts, model_path='models/trained/model.pkl',
                  model_name='distilbert-base-uncased', batch_size=32, threshold=0.5):
    """
    Quick prediction function for multiple texts.
    
    Args:
        texts: List of text strings
        model_path: Path to saved model
        model_name: HuggingFace model identifier
        batch_size: Batch size for processing
        threshold: Decision threshold
        
    Returns:
        List of prediction dictionaries
    """
    predictor = SentimentPredictor(model_path, model_name)
    predictor.load_model()
    return predictor.predict_batch(texts, batch_size, threshold)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This movie was amazing! I loved every minute of it.",
        "Terrible film. Waste of time.",
        "It was okay, nothing special."
    ]
    
    print("Running inference on sample texts...")
    results = predict_batch(sample_texts)
    
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
