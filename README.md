# SAN_Project — Robot Production Pipeline

The **SAN_Project** is a research-grade robotic intelligence system developed at the **Robotics Socially Aware Navigation Lab (RSAN Lab)**, *Sonoma State University (CS470)*.  
It integrates **computer vision (YOLOv9)**, **LLM reasoning (ChatGPT-Vision)**, and **ROS2 navigation** into a unified pipeline that enables robots to understand, interpret, and navigate socially aware environments.

---

## Overview

This project demonstrates the full **Robot Production Pipeline**, from **data collection** to **real-world robot deployment**.

<p align="center">
  <img src="docs/architecture_diagram.png" alt="SAN Project Pipeline" width="700">
</p>

Each stage of the pipeline is modular, allowing researchers to retrain, test, and integrate new AI or robotics components with minimal reconfiguration.

---

## Pipeline Stages

### 1. Data Collection & Cleaning
- Gathers raw image datasets (e.g., **COCO**, **SUN RGB-D**).  
- Cleans, normalizes, and augments data for consistent YOLO-ready format.  
- **Implemented in:**  
  - `colab/preprocess_data.ipynb`  
  - `src/perception/preprocess_data.py`

 *Why:* Preprocessing ensures balanced, high-quality inputs for model training and reduces noise in object detection performance.

---

### 2. Model Training (YOLOv9)
- Trains or fine-tunes YOLOv9 for **object detection** and **social scene understanding**.  
- Supports human-object interaction datasets.  
- **Implemented in:**  
  - `colab/train_yolov9.ipynb`  
  - `src/perception/train_yolo.py`

📈 *Why:* This step produces the robot’s perception model — its “eyes.”  
Fine-tuned weights are stored in:  
`models/yolov9_finetuned.pt`

---

### 3. Detection & Validation
- Loads the trained YOLO model for image or video inference.  
- Validates detections using FiftyOne or OpenCV visualization.  
- **Implemented in:**  
  - `colab/detect_yolov9.ipynb`  
  - `src/perception/detect_objects.py`

*Why:* Ensures the model performs correctly before integration into reasoning and navigation.

---

### 4. Reasoning (LLM + Rules)
- Combines YOLO detections with **language-based reasoning (ChatGPT-Vision)**.  
- Interprets scene semantics (“A person standing near a chair” → “occupied area”).  
- Logical decision-making handled via a **Rule Engine**.  
- **Implemented in:**  
  - `src/reasoning/llm_reasoner.py`  
  - `src/reasoning/rule_engine.py`

*Why:* This forms the “brain” — converting perception into semantic understanding for decision-making.

---

### 5. Navigation & Deployment (ROS2)
- Interfaces with **ROS2 (Humble)** for real-world robot deployment.  
- Translates reasoning outputs into motion commands.  
- Includes behavior scripts for autonomous decision-making.  
- **Implemented in:**  
  - `src/navigation/ros2_interface.py`  
  - `src/navigation/san_behavior.py`  
  - `ros_ws/src/san_node/san_node.py`

*Why:* Connects AI reasoning to robot motion — enabling context-aware navigation.

---

### 6. Evaluation & Documentation
- Logs model performance, reasoning outcomes, and ROS2 control responses.  
- Generates research-ready figures and markdown documentation.  
- **Implemented in:**  
  - `results/logs/`  
  - `docs/model_training.md`  
  - `docs/data_preparation.md`

*Why:* Enables reproducible experiments and data-driven improvement tracking.

---

## Directory Structure

