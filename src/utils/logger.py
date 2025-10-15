"""
Logging Configuration Module

This module sets up the logging system using loguru for better formatting
and management of log files.

Author: Bimidu Gunathilake
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional
from config import config


def setup_logger(
    log_file: Optional[str] = None,
    level: str = None,
    rotation: str = None,
    retention: str = None
) -> None:
    """
    Configure the logging system with file and console output.
    
    This function sets up loguru logger with:
    - Colored console output
    - Rotating file output
    - Custom format from configuration
    
    Args:
        log_file: Path to log file (default: logs/asd_detection.log)
        level: Logging level (default: from config)
        rotation: When to rotate log file (default: from config)
        retention: How long to keep old logs (default: from config)
    
    Example:
        >>> setup_logger()  # Use defaults from config
        >>> setup_logger(log_file="custom.log", level="DEBUG")
    """
    # Remove default logger
    logger.remove()
    
    # Set defaults from config
    level = level or config.logging.level
    rotation = rotation or config.logging.rotation
    retention = retention or config.logging.retention
    log_file = log_file or str(config.paths.logs_dir / "asd_detection.log")
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format=config.logging.format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler with rotation
    logger.add(
        log_file,
        format=config.logging.format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",  # Compress old logs
        backtrace=True,
        diagnose=True,
    )
    
    logger.info(f"Logger initialized - Level: {level}, File: {log_file}")


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger (typically __name__)
    
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logger.bind(name=name)


# Initialize logger on module import
setup_logger()


__all__ = ["setup_logger", "get_logger", "logger"]

