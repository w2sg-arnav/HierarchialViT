# hvit/utils/logging_setup.py
"""Logging configuration utilities.

This module provides utilities for setting up consistent logging
across the hvit package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Configure logging for the hvit package.
    
    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file.
        log_format: Optional custom log format string.
    
    Returns:
        Configured root logger for hvit package.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get or create hvit logger
    logger = logging.getLogger("hvit")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


__all__ = ["setup_logging"]
