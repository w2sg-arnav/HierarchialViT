# phase4_finetuning/utils/logging_setup.py
import os
import logging
from datetime import datetime
import sys
from typing import Optional 

def setup_logging(log_file_name: Optional[str] = None, 
                  log_dir: str = "logs", 
                  log_level: int = logging.INFO,
                  logger_name: Optional[str] = None):
    """
    Configure logging to a file and console. Creates log_dir if needed.
    Args:
        log_file_name (str, optional): Base name for log file. Timestamp appended if None.
        log_dir (str): Directory for logs.
        log_level (int): Logging level.
        logger_name (Optional[str]): Name of the logger to configure (None for root).
    """
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
             print(f"Error creating log directory {log_dir}: {e}")
             # Fallback log directory maybe? or raise error? For now, just print.
             log_dir = "." # Log in current directory as fallback

    if log_file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"run_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file_name)
    
    logger_instance = logging.getLogger(logger_name) # Get root logger if name is None
    logger_instance.setLevel(log_level) # Set level on the specific logger

    # Prevent adding handlers multiple times
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(log_path, mode='a') # Append mode
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
        logger_instance.info(f"Logging configured. Log file: {log_path}")
    except Exception as e:
        logger_instance.error(f"Failed to create file handler for {log_path}: {e}")

    # Prevent log propagation to root logger if configuring a specific logger
    # to avoid duplicate messages if root logger also has handlers.
    # However, often configuring the root logger (logger_name=None) is sufficient.
    if logger_name:
         logger_instance.propagate = False

    return logger_instance # Return the configured logger instance