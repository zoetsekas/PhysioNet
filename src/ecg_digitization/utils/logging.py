"""Logging setup using loguru."""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Setup logging with loguru."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_path / "train_{time}.log", rotation="100 MB", level=level)
    
    return logger
