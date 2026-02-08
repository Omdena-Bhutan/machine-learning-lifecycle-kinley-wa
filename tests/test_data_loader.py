import pytest
import pandas as pd
import os
from pathlib import Path
from src.data_loader import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """TODO: Create sample data for testing"""
        # TODO: Create a sample CSV with test data
        data = {
            'text': [
                'This movie was great',
                'I hated this film',
                'It was okay'
            ],
            'label': [1, 0, 1]
        }
        df = pd.DataFrame(data)
        return df
    
    @pytest.fixture
    def preprocessor(self, tmp_path):
        """TODO: Create DataPreprocessor instance with temporary directory"""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'text': ['Great movie', 'Bad film', 'Okay'],
            'label': [1, 0, 1]
        })
        df.to_csv(csv_path, index=False)
        
        return DataPreprocessor(str(csv_path)), tmp_path
    
    def test_load_data(self, preprocessor):
        """TODO: Test data loading functionality"""
        preprocessor_obj, _ = preprocessor
        # TODO: Assert data is loaded correctly
        # df = preprocessor_obj.load_data()
        # assert df is not None
        # assert len(df) == 3
        pass
    
    def test_missing_file(self):
        """TODO: Test error handling for missing data file"""
        # TODO: Assert FileNotFoundError is raised
        # with pytest.raises(FileNotFoundError):
        #     preprocessor = DataPreprocessor('nonexistent.csv')
        #     preprocessor.load_data()
        pass
    
    def test_preprocess(self, preprocessor):
        """TODO: Test text preprocessing"""
        preprocessor_obj, _ = preprocessor
        # TODO: Load and preprocess data
        # preprocessor_obj.load_data()
        # df = preprocessor_obj.preprocess()
        # assert df is not None
        # Check that text is lowercased
        # assert all(text.islower() for text in df['text'].str[0:1])
        pass
    
    def test_create_splits(self, preprocessor):
        """TODO: Test train/test split creation"""
        preprocessor_obj, _ = preprocessor
        # TODO: Create splits and validate
        # preprocessor_obj.load_data()
        # preprocessor_obj.preprocess()
        # train, test = preprocessor_obj.create_splits()
        # assert len(train) + len(test) == len(preprocessor_obj.df)
        pass
    
    def test_save_splits(self, preprocessor):
        """TODO: Test saving processed data"""
        preprocessor_obj, tmp_path = preprocessor
        # TODO: Save and verify files created
        # preprocessor_obj.load_data()
        # preprocessor_obj.preprocess()
        # preprocessor_obj.create_splits()
        # output_dir = str(tmp_path / "processed")
        # preprocessor_obj.save_splits(output_dir)
        # assert (Path(output_dir) / 'train.pkl').exists()
        # assert (Path(output_dir) / 'test.pkl').exists()
        pass
    
    def test_pipeline_end_to_end(self, preprocessor):
        """TODO: Test complete pipeline execution"""
        preprocessor_obj, tmp_path = preprocessor
        # TODO: Run full pipeline and validate output
        # output_dir = str(tmp_path / "processed")
        # preprocessor_obj.run_pipeline(output_dir)
        # assert (Path(output_dir) / 'train.pkl').exists()
        # assert (Path(output_dir) / 'test.pkl').exists()
        pass
    
    def test_empty_data(self, tmp_path):
        """TODO: Test handling of empty dataset"""
        # TODO: Create empty CSV and test error handling
        # csv_path = tmp_path / "empty.csv"
        # pd.DataFrame(columns=['text', 'label']).to_csv(csv_path, index=False)
        # preprocessor = DataPreprocessor(str(csv_path))
        # TODO: Assert error or warning
        pass
