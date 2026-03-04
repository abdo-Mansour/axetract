"""Logging utilities for the AXEtract package."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration for the package.

    Args:
        level (int): Logging level, defaults to logging.INFO.
    """
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
