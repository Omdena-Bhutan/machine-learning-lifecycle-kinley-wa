"""
Utility functions for the sentiment analysis project.
"""

import json
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    TODO: Load YAML configuration file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary with configuration
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    TODO: Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save YAML file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    TODO: Load JSON configuration file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    TODO: Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_pickle(file_path: str) -> Any:
    """
    TODO: Load pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pickle.UnpicklingError as e:
        raise ValueError(f"Error unpickling: {e}")


def save_pickle(obj: Any, file_path: str) -> None:
    """
    TODO: Save object to pickle file.
    
    Args:
        obj: Object to pickle
        file_path: Path to save pickle file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def ensure_directory(dir_path: str) -> Path:
    """
    TODO: Ensure directory exists, create if needed.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    TODO: Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    TODO: Validate configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    return True


def batch_iterator(items: List[Any], batch_size: int):
    """
    TODO: Create batches from a list.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
