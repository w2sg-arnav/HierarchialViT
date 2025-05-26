# phase3_pretraining/utils/logging_setup.py
import logging
import os
import sys
from datetime import datetime
from typing import Optional # <--- ADD THIS LINE


def setup_logging(log_dir_abs_path: str, # Expect absolute path
                  log_file_name: str = "phase3_default.log",
                  log_level_file: int = logging.DEBUG,
                  log_level_console: int = logging.INFO,
                  run_timestamp: Optional[str] = None):
    """
    Sets up logging to both a file and the console. Ensures it's only done once.
    """
    root_logger = logging.getLogger() # Get the root logger

    # Check if handlers are already configured for the root logger
    if root_logger.hasHandlers() and any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        # Simple check: if a FileHandler exists, assume logging is set up.
        # A more robust check might involve specific handler names or flags.
        root_logger.info("Logging seems to be already configured. Skipping setup_logging.")
        return

    # Ensure log directory exists
    os.makedirs(log_dir_abs_path, exist_ok=True)

    # Add timestamp to log file name to avoid overwriting (optional)
    _run_timestamp = run_timestamp if run_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    final_log_file_name = f"{os.path.splitext(log_file_name)[0]}_{_run_timestamp}.log"
    log_file_path = os.path.join(log_dir_abs_path, final_log_file_name)

    root_logger.setLevel(min(log_level_file, log_level_console))

    # --- File Handler ---
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') # 'w' to overwrite for new run
    file_handler.setLevel(log_level_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_console)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s') # Simpler for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Suppress overly verbose library loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root_logger.info(f"Logging configured. File: {log_file_path} (Level: {logging.getLevelName(log_level_file)}), Console Level: {logging.getLevelName(log_level_console)}")