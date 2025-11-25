# Welcome to the Robotics Socially Aware Navigation Lab (RSAN Lab)

This organization was established to facilitate research and development in **socially aware robotic navigation**, combining **computer vision**, **machine learning**, and **robotic reasoning** to create context-aware autonomous systems.

The **RSAN_Project** is a research-driven system developed for **CS470: Advanced Software Design Project** at **Sonoma State University**.  
It integrates **AI perception**, **Large Language Models (LLMs)**, and **robotic navigation** using **ROS2** to enable socially aware and intelligent robot behavior.

Through a modular, research-driven design, the system integrates perception, reasoning, and navigation to enable robots to interpret human-centric environments and respond appropriately.  
This work exemplifies advanced concepts in **object detection, scene understanding, contextual inference, and behavior-based robotics**.

---

## System Overview

The SAN_Project Robot Production Pipeline is composed of interconnected modules, each serving a distinct purpose in the overall robot intelligence framework.

1. **Data Collection & Cleaning**  
   - Gathers and refines real-world or simulated datasets (COC0).  
   - Processes image data to remove noise and ensure balanced training samples.  
   - Implemented in:  
     - `colab/preprocess_data.ipynb`  
     - `src/perception/preprocess_data.py`

2. **Model Training (YOLOv8)**  
   - Fine-tunes YOLOv8 on custom datasets for human-object interaction recognition.  
   - Integrates social cues into object detection for contextual awareness.  
   - Implemented in:  
     - `colab/train_yolov8.ipynb`  
     - `src/perception/train_yolo.py`

3. **Visual Perception & Detection**  
   - Runs trained models to detect people, objects, and obstacles in the environment.  
   - Supports both static images and video streams for real-time inference.  
   - Implemented in:  
     - `src/perception/detect_objects.py`

4. **LLM-Based Reasoning (ChatGPT-Vision Integration)**  
   - Uses GPT models for scene interpretation and semantic reasoning.  
   - Converts visual detections into natural language context for decision-making.  
   - Implemented in:  
     - `src/reasoning/llm_reasoner.py`  
     - `src/reasoning/rule_engine.py`

5. **Navigation & Behavior Control (ROS2 Integration)**  
   - Translates high-level reasoning into low-level motion commands.  
   - Uses ROS2 topics to manage path planning and robot movement.  
   - Implemented in:  
     - `src/navigation/ros2_interface.py`  
     - `src/navigation/san_behavior.py`

6. **Evaluation & Documentation**  
   - Tracks model performance and robot behavior metrics.  
   - Produces reports, visualizations, and reproducible results.  
   - Implemented in:  
     - `docs/model_training.md`  
     - `results/logs/`  

---

## Research Motivation

Modern robotics must extend beyond navigation — it must understand and adapt to **social contexts**.  
The **RSAN_Project** explores the fusion of **visual perception** and **language-based reasoning** to empower robots to:

- Recognize people, gestures, and environments  
- Understand contextual meaning in scenes  
- Navigate safely and respectfully in human spaces  

This layered design mirrors how humans combine **sight, reasoning, and decision-making** — a key step toward socially intelligent autonomous systems.

---

## Technologies and Frameworks

| Domain | Tool / Framework | Purpose |
|--------|------------------|----------|
| Computer Vision | **YOLOv8 / OpenCV** | Object detection and real-time inference |
| Dataset Management | **FiftyOne / Albumentations** | Data cleaning, augmentation, and visualization |
| Reasoning | **ChatGPT-Vision / LangChain** | Semantic interpretation of detected scenes |
| Robotics | **ROS2 (Humble)** | Real-world control and sensor integration |
| ML Workflow | **Python, PyTorch, Jupyter, Colab** | Model training and experimentation |

---

## Project Architecture
Project architecture here later 

---

## Installation & Setup Guide

To ensure all team members can reproduce the same environment on **Mac, Linux, or Windows**, follow one of the setup methods below.

---

### Recommended: Smart Environment Setup (Automatic)

This is the **preferred method** for all contributors.  
It automatically detects whether **Conda** is available and uses it for environment creation.  
If Conda isn’t found, it safely falls back to a **Python virtual environment (venv)**.

<!-- ```bash -->
# Clone the repository
git clone https://github.com/Robotics-Socially-Aware-Navigation-Lab/RSAN_Project.git
cd RSAN_Project

# Make the setup script executable
chmod +x setup.sh

# Run the smart setup (auto-detects Conda or Python venv)
bash setup.sh

### For Mac / Linux Users (Manual Setup)

# Clone the repository
git clone https://github.com/Robotics-Socially-Aware-Navigation-Lab/RSAN_Project.git
cd RSAN_Project

# Create environment from YAML
conda env create -f environment.yml

# Activate environment
conda activate rsan_env

# (Optional) Sync dependencies for pip users
# Update pip environment
pip freeze > requirements.txt

# Update conda environment
conda env export > environment.yml

## For Windows Users

# Clone the repository
git clone https://github.com/Robotics-Socially-Aware-Navigation-Lab/RSAN_Project.git
cd RSAN_Project

# Run setup (automatically configures the environment)
setup.bat

# Activate environment
call env\Scripts\activate

# To deactivate
deactivate

## After Installation

# Activate (Mac/Linux)
conda activate rsan_env
# or 
source env/bin/activate

# for Window
call env\Scripts\activat

# Deactivate
conda deactivate
# or
deactivate

### How to Run & Test Every Component of the RSAN System
#Before running any test:

cd ~/Desktop/RSAN_Project
conda activate rsan_env

# YOLOv8 Object Detection
python test_detection.py

# Detect Image or Video
python test_detect_any.py /path/to/image.jpg

python test_detect_any.py /path/to/video.mp4

## Webcam Detection
python test_detect_any.py webcam

### Indoor Scene / Room Classification (YOLOv8-CLS)

# Show Tool Options
python -m src.tools.run_scene_classification --help

# Classify Image
python -m src.tools.run_scene_classification photo.jpg --out-dir outputs/classification

# Classify Folder

python -m src.tools.run_scene_classification images_folder/

# Outputs:
outputs/classification/images/
outputs/classification/results.json

### Full Unified Pipeline

# Single Image
python -m src.tools.run_unified_pipeline path/to/image.jpg

# Video
python -m src.tools.run_unified_pipeline path/to/video.mp4

#  Webcam
python -m src.tools.run_unified_pipeline webcam

# Outputs:
outputs/full_pipeline/images/
outputs/full_pipeline/videos/
outputs/full_pipeline/logs/results.json