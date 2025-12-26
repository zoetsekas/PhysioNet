"""Logging setup using standard logging."""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Setup logging with standard library."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path / "train.log")
        ]
    )
    
    return logging.getLogger()
