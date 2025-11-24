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
Simple logging utility for the RSAN Project.

Provides a get_logger(name) function that returns a configured logger
with consistent formatting and INFO-level default logging.
"""

import logging
from pathlib import Path
from typing import Optional

# Internal cache to avoid duplicate logger creation
_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve or create a named logger with consistent formatting.

    Args:
        name (str | None): Optional logger name. If None, uses "rsan".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger_name = name if name else "rsan"

    # Return cached logger if it already exists
    if logger_name in _LOGGERS:
        return _LOGGERS[logger_name]

    # Create new logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Only add handlers if logger is new (prevents duplicates)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (outputs/logs/rsan.log)
        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "rsan.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prevent double propagation to root logger
        logger.propagate = False

    # Cache and return
    _LOGGERS[logger_name] = logger
    return logger
