# phase4_finetuning/utils/logging_setup.py
import os
import logging
from datetime import datetime # Make sure datetime is imported
import sys
from typing import Optional

def setup_logging(log_file_name: Optional[str] = None,
                  log_dir: str = "logs",
                  log_level: int = logging.INFO, # Overall level for handlers
                  logger_name: Optional[str] = None, # Target logger (None for root)
                  run_timestamp: Optional[str] = None): # Timestamp for filename
    """
    Configure logging to a file and console. Creates log_dir if needed.
    """
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
             print(f"ERROR (logging_setup): Error creating log directory {log_dir}: {e}", file=sys.stderr)
             log_dir = "."

    _timestamp_to_use = run_timestamp if run_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_file_name:
        _base_name = os.path.splitext(log_file_name)[0]
        _ext = os.path.splitext(log_file_name)[1] if os.path.splitext(log_file_name)[1] else ".log"
        # Ensure timestamp is part of the name if not already templated
        if _timestamp_to_use not in _base_name:
            final_log_file_name = f"{_base_name}_{_timestamp_to_use}{_ext}"
        else: # Assume timestamp is already correctly in log_file_name
            final_log_file_name = log_file_name
    else:
        final_log_file_name = f"run_{_timestamp_to_use}.log"
    
    log_path = os.path.join(log_dir, final_log_file_name)
    
    # Get the specific logger or the root logger
    # This is the logger instance whose handlers we will configure.
    logger_to_configure = logging.getLogger(logger_name)
    logger_to_configure.setLevel(log_level) # Set its effective level

    # Clear existing handlers from THIS logger_to_configure to prevent duplicates if called multiple times
    if logger_to_configure.hasHandlers():
        # print(f"DEBUG: Clearing existing handlers from logger '{logger_name if logger_name else 'root'}'.")
        for handler in logger_to_configure.handlers[:]:
            logger_to_configure.removeHandler(handler)
            handler.close()
    
    # If configuring the root logger, also ensure no other handlers are on it from previous basicConfigs
    if logger_name is None and logging.getLogger().handlers: # Check root logger specifically
        # print("DEBUG: Clearing root logger handlers specifically.")
        for handler in logging.getLogger().handlers[:]:
             logging.getLogger().removeHandler(handler)
             handler.close()


    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level) # Console gets the overall specified level
    logger_to_configure.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(log_path, mode='a') # Append mode
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level) # File also gets the overall specified level
                                         # (can be set more verbose, e.g., logging.DEBUG, if log_level_file param was used)
        logger_to_configure.addHandler(file_handler)
        # Log initial message using the configured logger instance
        logger_to_configure.info(f"Logging configured. Log file: {log_path}. Logger '{logger_name if logger_name else 'root'}' Effective Level: {logging.getLevelName(logger_to_configure.getEffectiveLevel())}")
    except Exception as e:
        # If file handler fails, console handler should still work.
        logger_to_configure.error(f"Failed to create file handler for {log_path}: {e}")

    # Suppress overly verbose library loggers if desired (on the root logger)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if logger_name:
         logger_to_configure.propagate = False # Prevent messages from this named logger going to root
    
    return logger_to_configure