"""Logging setup for ForecastOps."""

import logging
import os
from typing import Optional


def setup_logger(
    name: str = "forecastops",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Set up and return a logger instance.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or default
    level_str = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(log_format)
        logger.addHandler(ch)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    
    return logger


# Default logger
logger = setup_logger()
