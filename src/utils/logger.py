# import logging
# from datetime import datetime
# from pathlib import Path

# def setup_logger(log_dir="results/logs", name="san_project"):
#     """Create and configure a logger that writes both to console and file."""
#     Path(log_dir).mkdir(parents=True, exist_ok=True)
#     log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(name)

"""
logger.py
Purpose:
    Standardized logger for all modules.
    Each log is saved in a timestamped file under results/logs/
    and mirrored in the console output.

How:
    - Uses Python's logging module
    - Writes both to console and file
    - Includes timestamps and log levels (INFO, ERROR)

Why:
    - Keeps experiment output organized
    - Makes debugging easier when running multiple Colab sessions or robots
"""

import logging
from datetime import datetime
from pathlib import Path

def setup_logger(log_dir="results/logs", name="san_project"):
    """WHAT: Configure and return a logger instance.
       HOW: Creates file and stream handlers to write logs to both console and file.
       WHY: So every training or preprocessing run is timestamped and traceable."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # writes to file
            logging.StreamHandler()         # prints to terminal
        ]
    )
    return logging.getLogger(name)