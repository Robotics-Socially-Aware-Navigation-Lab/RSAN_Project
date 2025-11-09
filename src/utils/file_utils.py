# import yaml, os, json
# from pathlib import Path

# def load_yaml(yaml_path):
#     """Load any YAML configuration file."""
#     with open(yaml_path, 'r') as f:
#         return yaml.safe_load(f)

# def load_paths(config_path='configs/project_paths.yaml'):
#     """Load project paths as a dictionary."""
#     cfg = load_yaml(config_path)
#     return {k: Path(v) for k, v in cfg['paths'].items()}

# def ensure_dirs(paths):
#     """Ensure all critical directories exist."""
#     for p in paths.values():
#         os.makedirs(p, exist_ok=True)

# def save_json(data, path):
#     """Save Python dict as JSON."""
#     with open(path, 'w') as f:
#         json.dump(data, f, indent=4)

# def load_json(path):
#     """Load JSON data."""
#     with open(path, 'r') as f:
#         return json.load(f)

"""
file_utils.py
Purpose:
    Central utility functions for file handling and configuration loading.
    Every major module (preprocessing, training, reasoning) depends on this file
    to access paths, configs, and save outputs consistently.

How:
    - Uses YAML for configurations
    - Ensures directories exist automatically
    - Provides helper functions for JSON reading/writing

Why:
    - Avoids hardcoded paths
    - Keeps file structure portable between Colab, local, and ROS environments
"""

import yaml, os, json
from pathlib import Path

def load_yaml(yaml_path):
    """WHAT: Load YAML file into Python dict.
       HOW: Uses safe YAML loader.
       WHY: Central way to access configuration files."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_paths(config_path='configs/project_paths.yaml'):
    """WHAT: Load project directory structure as a dictionary.
       HOW: Reads from project_paths.yaml.
       WHY: Makes file locations dynamic and consistent."""
    cfg = load_yaml(config_path)
    return {k: Path(v) for k, v in cfg['paths'].items()}

def ensure_dirs(paths):
    """WHAT: Create directories if they don't exist.
       HOW: Loops through dictionary of Path objects and creates them.
       WHY: Prevents file-not-found errors during data writes."""
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

def save_json(data, path):
    """WHAT: Save Python dict as a JSON file.
       HOW: Opens file in write mode and dumps formatted JSON.
       WHY: Standard format for detection results or metadata."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    """WHAT: Load data from a JSON file.
       HOW: Reads and parses JSON into a Python object.
       WHY: To reuse saved metadata, detections, or reasoning output."""
    with open(path, 'r') as f:
        return json.load(f)