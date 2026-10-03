"""
logging_config.py - Centralised logging configuration for Part 2 scripts.
"""

import logging
import os
import sys

LOG_DIR = os.path.join("capstone_part2", "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")


def setup_logging(debug: bool = False) -> None:
    """Configure root logger with console and file handlers.

    Parameters
    ----------
    debug : bool
        If True, set both handlers to DEBUG; otherwise INFO.
    """
    root = logging.getLogger()

    # Only configure once
    if root.handlers:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
