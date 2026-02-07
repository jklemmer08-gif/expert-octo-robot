"""Logging configuration for PPP Processor.

Provides rotating file handler + console output with structured format.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: Path | str = "logs",
    log_file: str = "ppp_processor.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the root PPP logger.

    Creates both a rotating file handler and a console handler with a
    structured format including timestamps and module names.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ppp")
    logger.setLevel(level)

    # Avoid duplicate handlers on repeat calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = RotatingFileHandler(
        str(log_path / log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ppp namespace."""
    return logging.getLogger(f"ppp.{name}")
