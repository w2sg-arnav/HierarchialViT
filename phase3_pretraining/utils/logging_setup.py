# phase3_pretraining/utils/logging_setup.py
import logging
import os
import sys
from datetime import datetime

def setup_logging(log_file_name: str = "default.log",
                  log_dir: str = "logs",
                  log_level_file: int = logging.DEBUG,
                  log_level_console: int = logging.INFO, # INFO for console
                  package_name: str = "phase3_pretraining"):
    """
    Sets up logging to both a file and the console.

    Args:
        log_file_name (str): Name of the log file.
        log_dir (str): Directory to store log files.
        log_level_file (int): Logging level for the file handler.
        log_level_console (int): Logging level for the console handler.
        package_name (str): The root package name for filtering/formatting if needed.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Use a timestamp in the log file name to avoid overwriting for different runs
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # full_log_file_name = f"{os.path.splitext(log_file_name)[0]}_{timestamp}{os.path.splitext(log_file_name)[1]}"
    # For simplicity with your current config, let's use the provided log_file_name directly
    full_log_file_name = log_file_name
    log_file_path = os.path.join(log_dir, full_log_file_name)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level_file, log_level_console)) # Set to the lower of the two

    # --- File Handler ---
    # Create file handler
    # Use 'a' to append if file exists, useful for long runs or resuming
    # file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    # For pretraining, often 'w' is preferred to start fresh for each run
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level_file)

    # Create formatter and add it to the handlers
    # Example: [2023-10-27 10:00:00,123 - phase3_pretraining.module - INFO] - Message
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for console
    console_handler.setLevel(log_level_console)

    # More concise formatter for console
    # Example: [INFO] [module] Message
    console_formatter = logging.Formatter(
        '[%(levelname)s] [%(name)s.%(funcName)s] %(message)s'
    )
    # Or even simpler for just level and message:
    # console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add the handlers to the root logger
    # Clear existing handlers first to avoid duplicate logging if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Optionally, suppress overly verbose loggers from libraries
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.INFO) # Suppress PIL DEBUG
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


    root_logger.info(f"Logging setup complete. File: {log_file_path} (Level: {logging.getLevelName(log_level_file)}), Console Level: {logging.getLevelName(log_level_console)}")