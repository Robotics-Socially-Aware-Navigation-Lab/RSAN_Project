
# """
# logger.py
# Purpose:
#     Standardized logger for all modules.
#     Each log is saved in a timestamped file under results/logs/
#     and mirrored in the console output.

# How:
#     - Uses Python's logging module
#     - Writes both to console and file
#     - Includes timestamps and log levels (INFO, ERROR)

# Why:
#     - Keeps experiment output organized
#     - Makes debugging easier when running multiple Colab sessions or robots
# """

# import logging
# from datetime import datetime
# from pathlib import Path


# def setup_logger(log_dir="results/logs", name="san_project"):
#     """WHAT: Configure and return a logger instance.
#     HOW: Creates file and stream handlers to write logs to both console and file.
#     WHY: So every training or preprocessing run is timestamped and traceable."""

#     Path(log_dir).mkdir(parents=True, exist_ok=True)
#     log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[logging.FileHandler(log_file), logging.StreamHandler()],  # writes to file  # prints to terminal
#     )
#     return logging.getLogger(name)

"""
Simple logging utility for RSAN Project.

Provides a get_logger(name) function that returns a module-level logger
with consistent formatting and INFO-level default logging.
"""

import logging
from pathlib import Path
from typing import Optional

_LOGGERS = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    global _LOGGERS
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name if name else "rsan")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional file handler under outputs/logs/
        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "rsan.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.propagate = False

    _LOGGERS[name] = logger
    return logger