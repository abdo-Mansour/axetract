"""Logging utilities for the AXEtract package."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path = "axetract.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Setup logging configuration for the package.

    Logs are written to stdout. A file handler is only added if debug mode is on.
    Debug mode can be triggered via the `AXE_DEBUG=1` environment variable.

    Args:
        level (int): Logging level, defaults to logging.INFO. Overridden by AXE_DEBUG.
        log_file (str | Path): Path to the log file. Defaults to ``axetract.log``
            in the current working directory.
        max_bytes (int): Maximum size in bytes before the log file rotates.
            Defaults to 10 MB.
        backup_count (int): Number of rotated backup files to keep.
            Defaults to 5.
    """
    is_debug_env = os.environ.get("AXE_DEBUG", "0").lower() in ("1", "true", "yes")
    if is_debug_env:
        level = logging.DEBUG

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicates if called multiple times
    root_logger.handlers.clear()

    root_logger.setLevel(level)

    # stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # optionally add file handler
    is_debug_mode = (level <= logging.DEBUG)
    if is_debug_mode:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

