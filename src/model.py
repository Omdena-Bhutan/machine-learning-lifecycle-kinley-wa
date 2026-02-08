import os
import json
import pickle
import yaml
import mlflow
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset


class SentimentModel:
    """
    Fine-tunes a pre-trained transformer model for sentiment analysis.
    """
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        """
        Initialize the sentiment model.
        
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of classification labels (2 for binary sentiment)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def load_model_and_tokenizer(self):
        """Load pre-trained tokenizer and model from HuggingFace"""
        print(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        print("Model and tokenizer loaded successfully")
        return self.model, self.tokenizer
    
    def freeze_base_layers(self, freeze=True):
        """Freeze base model layers for transfer learning"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        print(f"Freezing base layers: {freeze}")
        
        if freeze:
            for param in self.model.distilbert.parameters():
                param.requires_grad = False
        
        # Ensure classifier layers are trainable
        for param in self.model.pre_classifier.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def load_data(self, train_path, max_length=128):
        """Load and tokenize training data"""
        print(f"Loading training data from {train_path}")
        
        with open(train_path, 'rb') as f:
            train_df = pickle.load(f)
        
        train_dataset = Dataset.from_pandas(train_df)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                max_length=max_length,
                truncation=True,
                padding=True
            )
        
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def train(self, train_path, val_split=0.2, config_path='params.yaml'):
        """Train the model with MLflow logging"""
        
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        train_params = params['train']
        
        print(f"Starting training with parameters: {train_params}")
        
        if self.model is None:
            self.load_model_and_tokenizer()
        
        train_dataset = self.load_data(train_path, train_params['max_length'])
        
        with mlflow.start_run():
            mlflow.log_params({
                'model_name': self.model_name,
                'batch_size': train_params['batch_size'],
                'epochs': train_params['epochs'],
                'learning_rate': train_params['learning_rate'],
                'max_length': train_params['max_length']
            })
            
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=train_params['epochs'],
                per_device_train_batch_size=train_params['batch_size'],
                per_device_eval_batch_size=train_params['batch_size'],
                warmup_steps=train_params.get('warmup_steps', 500),
                weight_decay=train_params.get('weight_decay', 0.01),
                learning_rate=train_params['learning_rate'],
                logging_dir='./logs',
                logging_steps=100,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='accuracy'
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=DataCollatorWithPadding(self.tokenizer),
            )
            
            train_result = self.trainer.train()
            
            mlflow.log_metrics({
                'train_loss': train_result.training_loss,
            })
            
            self.save_model()
            
            print("Training completed!")
    
    def save_model(self, output_dir='models/trained'):
        """Save model and tokenizer to disk"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"Saving model to {output_dir}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Initialize and train model
    model = SentimentModel(
        model_name=params['train']['model_name'],
        num_labels=2
    )
    
    model.train(
        train_path='data/processed/train.pkl',
        val_split=params['train']['validation_split']
    )
