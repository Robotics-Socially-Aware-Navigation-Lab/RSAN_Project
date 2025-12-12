# Welcome to the Robotics Socially Aware Navigation Lab (RSAN Lab)

This group was founded to advance research in **socially aware robotic perception**, combining **computer vision**, **indoor scene understanding**, 
and **AI reasoning** to help robots behave intelligently in human environments.

The **RSAN_Project** was developed for **CS470: Advanced Software Design Project** at **Sonoma State University**.  
It integrates **YOLOv8 indoor object detection**, **indoor scene classification**, **hybrid fusion**, and **LLM reasoning**, producing a unified perception pipeline capable of interpreting complex indoor spaces.

The system is fully modular and research-oriented, enabling:

- Indoor object detection  
- Indoor room classification  
- Hybrid scene refinement  
- Symbolic + LLM reasoning  
- Image, video, and webcam-based real-time perception  

---

# System Overview (RSAN Perception Pipeline)

The RSAN pipeline consists of tightly integrated modules that work together from **data → trained models → reasoning → annotated outputs**.

---

## 1. Data Collection & Preprocessing

We collect and prepare data for **indoor object detection** and **indoor scene classification**.

**Object Detection datasets:**
- COCO (filtered to indoor classes)  
- HomeObject3K  
- Roboflow indoor datasets  

**Scene Classification datasets:**
- MIT Indoor Scenes  
- Places365  

After filtering indoor classes, we preprocess the data into YOLO-ready format.

**Implemented in:**
- `colab/preprocess_data.ipynb`  
- `src/perception/preprocess_data.py`

---

## 2. Model Training

### YOLOv8 Indoor Object Detection  
Trains YOLOv8 on indoor-specific object classes (chair, bed, toilet, laptop, couch, etc.).

Outputs include:
- `best.pt`  
- Training logs  
- Confusion matrix  
- Validation results  

### Indoor Scene Classification (ResNet-Places365 Multi-Head)  
Your real indoor classifier:

### resnet_places365_best.pth

This model contains:
- ResNet50 backbone  
- Places365 head  
- MIT indoor scene classification head (used by RSAN)  

**Implemented in:**
- `colab/train_yolov8.ipynb`  
- `colab/Places365_Detecting_Videos.ipynb`  
- `src/reasoning/indoor_classifier.py`

---

## 3. Visual Perception (Object Detection + Scene Classification)

### Indoor Object Detection  
Runs YOLOv8 to detect indoor objects.

Implemented in:
- `src/perception/detect_objects.py`

### Indoor Scene Classification  
Runs in our ResNet multi-head classifier to classify indoor room type.

Implemented in:
- `src/reasoning/indoor_classifier.py`

---

## 4. Hybrid Scene Fusion (RSAN Innovation)

RSAN refines indoor scene classification using detected objects.

**Example:**  
If the classifier predicts **“office”**, but YOLO detects **bed + pillow**,  
Hybrid Fusion corrects it to **“bedroom.”**

Implemented in:
- `src/reasoning/scene_context.py`

---

## 5. Reasoning Layer (Symbolic + LLM)

Two reasoning systems work together:

### Rule-Based Reasoning  
Produces short semantic summaries based on objects + room classification.  
Used for:
- Images  
- Videos  
- Webcam HUD  

### LLM Reasoning (GPT-4o-Mini)  
Used only when an API key is present.  
Generates detailed natural-language understanding of the scene.

Implemented in:
- `src/reasoning/rule_engine.py`  
- `src/reasoning/llm_reasoner.py`  

Outputs stored in:
- `outputs/reasoning_output.txt`  
- `outputs/full_pipeline/logs/results.json`  

---

## 6. Full Unified Pipeline

Combines:
- YOLOv8 object detection  
- Indoor classifier  
- Hybrid fusion  
- Rule-based + LLM reasoning  
- Annotated overlays  

Implemented in:
- `src/tools/run_unified_pipeline.py`

Supports:
- Images  
- Videos  
- Webcam  

Outputs stored in: The outputs files, images, videos and result.json

---

## Research Motivation

Modern robotics must extend beyond navigation — it must understand and adapt to **social contexts**.  
The **RSAN_Project** explores the fusion of **visual perception** and **language-based reasoning** to empower robots to:

- Recognize people, gestures, and environments  
- Understand contextual meaning in scenes  
- Navigate safely and respectfully in human spaces  

This layered design mirrors how humans combine **sight, reasoning, and decision-making** — a key step toward socially intelligent autonomous systems.

---

# Technologies Used

| Area | Tools |
|------|-------|
| Object Detection | YOLOv8 + OpenCV |
| Indoor Scene Classification | ResNet-Places365 Multi-Head |
| Hybrid Fusion | Custom Python logic |
| Reasoning | Rule-Based + GPT-4o-Mini |
| ML & Training | PyTorch, Jupyter, Colab |
| Environment | Conda, pip, environment.yml |

---
## Project Contributors:
## Professor:
- Dr Gurman Gill
## Stakeholder:
- Dr. Salek and her research team
## Students:
- Rolando Yax  
- Kyle Garrity  
- Jonathan Ramirez  
- Gurkirat Sandhu

---

# Installation & Setup

## Recommended: Smart Auto-Setup

```bash
git clone https://github.com/Robotics-Socially-Aware-Navigation-Lab/RSAN_Project.git
cd RSAN_Project

chmod +x setup.sh
bash setup.sh

## Manual Setup (Mac / Linux)
git clone https://github.com/Robotics-Socially-Aware-Navigation-Lab/RSAN_Project.git
cd RSAN_Project

conda env create -f environment.yml
conda activate rsan_env

## Windows Setup
git clone https://github.com/Robotics-Socially-Aware-Navigation-Lab/RSAN_Project.git
cd RSAN_Project

setup.bat
call conda activate rsan_env


### Running & Testing RSAN
> Run all commands from the project root (`RSAN_Project`).
> Paths can be **relative** or **absolute**.

conda activate rsan_env


## YOLOv8 Object Detection

# Single Image
python test_detection.py path/to/image.jpg

# Example
python test_detection.py data/images/room.jpg

# Single Image (Generic Detector)
python test_detect_any.py path/to/image.jpg

# Example
python test_detect_any.py data/images/office.jpg

# Video
python test_detect_any.py path/to/video.mp4

# Example
python test_detect_any.py data/videos/indoor_scene.mp4

# Webcam
python test_detect_any.py webcam

---

## Indoor Scene Classification (ResNet-MIT-Places365)

# Single Image
python -m src.tools.run_scene_classification path/to/image.jpg --out-dir outputs/classification

# Example
python -m src.tools.run_scene_classification data/images/classroom.jpg --out-dir outputs/classification

# Folder of Images
python -m src.tools.run_scene_classification path/to/images_folder/

# Example
python -m src.tools.run_scene_classification data/images/

# Outputs saved to:
outputs/classification/images/
outputs/classification/results.json

---

## Full Unified Perception Pipeline

# Single Image
python -m src.tools.run_unified_pipeline path/to/image.jpg

# Example
python -m src.tools.run_unified_pipeline data/images/room.jpg

---

# Folder of Images
python -m src.tools.run_unified_pipeline path/to/images_folder/

# Example
python -m src.tools.run_unified_pipeline data/images/

---

# Single Video
python -m src.tools.run_unified_pipeline path/to/video.mp4

# Example
python -m src.tools.run_unified_pipeline data/videos/indoor_scene.mp4

---

# Folder of Videos
python -m src.tools.run_unified_pipeline path/to/videos_folder/

# Example
python -m src.tools.run_unified_pipeline data/videos/

---

# Webcam
python -m src.tools.run_unified_pipeline webcam

---

# Outputs saved to:
outputs/full_pipeline/images/
outputs/full_pipeline/videos/
outputs/full_pipeline/logs/results.json

## Code Quality (REQUIRED Before Every Push)
black . && ruff check . --fix && flake8 src/
