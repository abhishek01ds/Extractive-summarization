"""
src/utils.py: Utility functions for SGCSumm.
Includes configuration loading, reproducible random seed setting, device management,
and unified logging.
"""

import os
import random
import logging
from typing import Any, Dict
import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file safely.
    
    Args:
        config_path: Path to configuration YAML file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for random, numpy, and torch to ensure full reproducibility.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cpu") -> torch.device:
    """
    Select and return torch.device based on user configuration and availability.
    
    Args:
        device_str: "cpu", "cuda", or "auto"
        
    Returns:
        torch.device object.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def setup_logger(
    name: str = "SGCSumm",
    log_file: str = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Configure and return a structured logger.
    
    Args:
        name: Logger name.
        log_file: Optional file path to write logs.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        
    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional File handler
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


def ensure_dir(directory: str) -> str:
    """Ensure directory exists; create if necessary."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory
