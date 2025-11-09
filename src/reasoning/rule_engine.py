"""
rule_engine.py
Purpose:
    Convert semantic scene understanding into logical or control actions.
    Example: "If person and door detected → open door."

How:
    - Reads textual reasoning results
    - Checks conditions with simple string or semantic matching
    - Logs and returns control commands

Why:
    - Introduces low-latency, interpretable decision-making layer
    - Essential for robot navigation and interaction
"""

from utils.logger import setup_logger

def apply_rules(reasoning_text):
    logger = setup_logger(name="rule_engine")

    if "person" in reasoning_text and "door" in reasoning_text:
        action = "Open the door"
    elif "no obstacles" in reasoning_text:
        action = "Move forward"
    else:
        action = "Stay idle"

    logger.info(f"🤖 Rule-based Action: {action}")
    return action