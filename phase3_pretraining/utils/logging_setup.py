# phase3_pretraining/utils/logging_setup.py
from typing import Optional, Union, Tuple
import logging
import sys
import os

def setup_logging(log_file_name: str = "default.log", 
                  log_level: int = logging.INFO, 
                  log_dir: str = "logs",
                  logger_name: Optional[str] = None):
    """
    Sets up logging to a file and to the console.
    Creates the log directory if it doesn't exist.
    Args:
        log_file_name (str): Name of the log file.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_dir (str): Directory to save log files.
        logger_name (Optional[str]): If None, configures the root logger. 
                                     Otherwise, configures the named logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    full_log_path = os.path.join(log_dir, log_file_name)

    handlers = [
        logging.FileHandler(full_log_path, mode='a'), # Append mode
        logging.StreamHandler(sys.stdout)
    ]

    # Get the specific logger or the root logger
    logger_to_configure = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    
    # Remove existing handlers from this logger to avoid duplicate logs if called multiple times
    if logger_to_configure.hasHandlers():
        logger_to_configure.handlers.clear()

    logger_to_configure.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger_to_configure.addHandler(handler)

    if logger_name:
        logger_to_configure.info(f"Logging setup for '{logger_name}' to {full_log_path}")
    else:
        logger_to_configure.info(f"Root logging setup to {full_log_path}")

from typing import Optional # Add this for Optional type hint