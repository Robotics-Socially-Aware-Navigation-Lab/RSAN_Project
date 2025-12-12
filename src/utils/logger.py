
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
