import pytest
from src.model import SentimentModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TestSentimentModel:
    """Test suite for SentimentModel class"""
    
    @pytest.fixture
    def model_instance(self):
        """TODO: Create SentimentModel instance"""
        model = SentimentModel(model_name="distilbert-base-uncased", num_labels=2)
        return model
    
    def test_initialization(self, model_instance):
        """TODO: Test model initialization"""
        # TODO: Assert model is properly initialized
        # assert model_instance.model_name == "distilbert-base-uncased"
        # assert model_instance.num_labels == 2
        # assert model_instance.tokenizer is None
        # assert model_instance.model is None
        pass
    
    def test_load_model_and_tokenizer(self, model_instance):
        """TODO: Test loading pre-trained model and tokenizer"""
        # TODO: Load model and validate
        # model, tokenizer = model_instance.load_model_and_tokenizer()
        # assert tokenizer is not None
        # assert model is not None
        # assert model.num_labels == 2
        pass
    
    def test_freeze_layers(self, model_instance):
        """TODO: Test layer freezing for transfer learning"""
        # TODO: Load model and freeze layers
        # model_instance.load_model_and_tokenizer()
        # model_instance.freeze_base_layers(freeze=True)
        
        # TODO: Check that base layers are frozen
        # for name, param in model_instance.model.distilbert.named_parameters():
        #     assert param.requires_grad == False, f"{name} should be frozen"
        
        # TODO: Check that classifier is trainable
        # for name, param in model_instance.model.classifier.named_parameters():
        #     assert param.requires_grad == True, f"{name} should be trainable"
        pass
    
    def test_invalid_model_name(self):
        """TODO: Test error handling for invalid model names"""
        # TODO: Assert error for non-existent model
        # model = SentimentModel(model_name="invalid-model-name")
        # with pytest.raises(OSError):
        #     model.load_model_and_tokenizer()
        pass
    
    def test_train_without_data(self, model_instance):
        """TODO: Test that training fails without data"""
        # TODO: Assert error when training data not provided
        # with pytest.raises(FileNotFoundError):
        #     model_instance.train('nonexistent.pkl')
        pass
    
    def test_save_model(self, model_instance, tmp_path):
        """TODO: Test model saving functionality"""
        # TODO: Load, train, and save model
        # model_instance.load_model_and_tokenizer()
        # output_dir = str(tmp_path)
        # model_instance.save_model(output_dir)
        # assert (Path(output_dir) / 'model.pkl').exists()
        pass
    
    def test_different_model_names(self):
        """TODO: Test initialization with different model names"""
        models = [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base"
        ]
        
        # TODO: Test each model initialization
        # for model_name in models:
        #     model = SentimentModel(model_name=model_name)
        #     assert model.model_name == model_name
        pass
