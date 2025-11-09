"""
llm_reasoner.py
Purpose:
    Use ChatGPT Vision (or similar LLM) to interpret YOLO detections semantically.
    Converts bounding box detections into natural language descriptions.

How:
    - Loads detections JSONs
    - Constructs scene description prompts
    - Calls OpenAI API (gpt-4o-mini or gpt-4-vision-preview)
    - Saves reasoning output text to /results/

Why:
    - Adds semantic reasoning on top of object detection
    - Bridges perception → cognition → decision-making for robots
"""

import os, json
from openai import OpenAI
from utils.file_utils import load_paths
from utils.logger import setup_logger

def llm_reason():
    paths = load_paths()
    logger = setup_logger(name="llm_reasoner")

    # Load detection JSON output
    detection_file = list((paths["detections"]).glob("*.json"))[0]
    detections = json.load(open(detection_file))

    # Build prompt for GPT-Vision reasoning
    prompt = f"Analyze the following detections for robotic scene understanding:\n{detections}"

    # Connect to OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    reasoning_output = response.choices[0].message.content
    logger.info(f"LLM Reasoning Output:\n{reasoning_output}")

    # Save reasoning results
    output_path = paths["results"] / "reasoning_output.txt"
    with open(output_path, "w") as f:
        f.write(reasoning_output)

if __name__ == "__main__":
    llm_reason()