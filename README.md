# ✋ Hand Marks Detection  
### Real-Time Hand Gesture Recognition Using Landmark-Based Machine Learning

---

## 1️⃣ Project Overview

Hand Marks Detection is a real-time computer vision system for hand gesture recognition built using landmark-based feature engineering and classical machine learning.

The system combines:

- MediaPipe Hand Landmarker for robust 3D hand landmark extraction  
- A Random Forest classification pipeline for gesture prediction  
- Multi-input support (Live Camera, Video, Image)  
- Temporal smoothing for stable real-time inference  

The project is designed for CPU-efficient deployment, high interpretability, and strong generalization across real-world variations such as rotation, finger spread differences, and slight landmark noise.

---

## 2️⃣ Dataset Description

The dataset consists of **21 hand landmarks per sample**, extracted using MediaPipe.

Each sample contains:

- 21 landmarks  
- Each landmark includes `(x, y, z)` coordinates  
- A corresponding gesture label  

<p align="center">
  <img src="https://github.com/user-attachments/assets/357692c0-4080-4f3d-aa5c-1801ac64cc51" width="60%">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/34e3284e-0eda-4e95-afa7-1f685d604836" width="60%">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/8ee9d9d3-5034-4f5d-976e-9eb28e430bec" width="60%">
</p>

### Dataset Statistics

- 20,540 training samples  
- Separate development and test splits  
- Multiple dataset variants for experimentation:
  - `hand_landmarks_data.csv`
  - `hand_landmarks_data_f32.csv`
  - `hand_landmarks_data_no_z.csv`

<p align="center">
  <img src="https://github.com/user-attachments/assets/99578bd7-8d68-48e3-bb95-87c1d75a8dcf" width="60%">
</p>

### Structure

```
dataset/
├── hand_landmarks_data.csv
└── splits/v1/
    ├── hand_landmarks_train.csv
    ├── hand_landmarks_dev.csv
    └── hand_landmarks_test.csv
```

The dataset was carefully split to:

- Prevent data leakage  
- Ensure robust generalization  
- Evaluate real-world deployment performance  

---


## 3️⃣ Preprocessing Pipeline

The project follows a structured preprocessing approach before model training.

### Landmark Extraction

- MediaPipe Hand Landmarker extracts 21 landmarks per hand  
- Each landmark includes `(x, y)` coordinates (z was optionally evaluated during experiments)  

---

### Feature Engineering

Raw landmark coordinates alone are not geometrically invariant.  
To improve generalization and robustness, we applied the following transformations:

#### 1️⃣ Re-centering (Translation Invariance)

All landmarks are re-centered relative to the wrist (landmark index 0):

- The wrist becomes the origin `(0, 0)`
- All other landmarks are translated accordingly

This removes dependency on absolute hand position in the image.

```
wrist = X[:, 0:1, :]
X = X - wrist
```

---

#### 2️⃣ Scale Normalization (Size Invariance)

To make the model invariant to hand size and camera distance:

- The distance between the wrist and the middle finger tip (landmark 12) is computed  
- All landmarks are divided by this scale factor  

```
middle_tip = X[:, 12, :]
scale = np.linalg.norm(middle_tip, axis=1).reshape(-1, 1, 1)
X = X / scale
```

This ensures that gestures are compared based on **relative geometry**, not absolute size.

Division-by-zero cases are safely handled to maintain numerical stability.

---

#### 3️⃣ Flattening

After geometric normalization:

- Landmarks are reshaped from `(21, 2)` to `(42,)`
- Each sample becomes a structured tabular feature vector  

Final feature shape:

```
(N, 42)
```

---

### Scaling & Model Pipeline

The final deployed model is wrapped inside a **Scikit-learn Pipeline**, which includes:

- The preprocessing transformation (re-centering + normalization)
- Random Forest classifier
- Label encoding via `LabelEncoder`

This ensures:

- Identical preprocessing during training and inference  
- No feature mismatch in production  
- Clean modular design  
- Reproducible deployment  

---

## 4️⃣ Model Selection & Research Findings

Several classical machine learning models were evaluated:

- Decision Tree  
- Random Forest  
- Support Vector Machine (RBF Kernel)  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  

All experiments were tracked using **MLflow** for reproducibility and experiment comparison.

---

### 🔎 Key Findings

#### 🌳 Decision Tree

- 100% training accuracy  
- Significant overfitting (~6% drop on test set)  
- Deep decision boundaries memorized patterns  

#### 🧠 Support Vector Machine (RBF)

- Stable generalization  
- Sensitive to hand rotation  
- Increased confusion between similar gestures ("peace" vs "two_up")  

#### 🌲 Random Forest (Selected Model)

- ~97.4% overall accuracy  
- Minimal overfitting (~2.5% gap)  
- Strong robustness to:
  - Rotation variations  
  - Finger spread differences  
  - Minor landmark noise  
- CPU efficient  
- Highly stable for real-time deployment  

---

## 🏆 Final Model: Random Forest

Random Forest was selected due to:

- Best performance across all metrics  
- Strong generalization  
- Better handling of geometric ambiguity  
- Suitability for real-time CPU inference  

The final deployed model is saved as:

```
models/random_forest_pipeline.pkl
```

---

## 5️⃣ Real-Time Inference Features

The system supports multiple input modes:

- 🎥 Live Camera Feed (mirrored for natural interaction)  
- 📹 Video File Processing  
- 🖼 Image Processing  

Additional runtime improvements:

- Temporal landmark smoothing  
- Prediction majority voting buffer  
- Dynamic frame resizing  
- Adaptive text rendering for all resolutions  

---

## 6️⃣ Project Structure

```
Hand Marks Detection/
├── main.py
├── requirements.txt
├── hand_landmarker.task
├── dataset/
├── models/
│   ├── random_forest_pipeline.pkl
│   └── transformers/
│       └── label_encoder.pkl
├── helpers/
└── notebooks/
```

---

## 7️⃣ Demo & Results

### 📷 Sample Image Predictions

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f6f34b6-c40c-444e-81e3-dc9f31451079" width="50%">
  <img src="https://github.com/user-attachments/assets/1627d4af-f282-43c9-b857-d21fd0caeb52" width="50%">
  <img src="https://github.com/user-attachments/assets/58828bd3-ddae-40db-9245-e1e9691d8840" width="50%">
  <img src="https://github.com/user-attachments/assets/6926bf10-a906-4610-ae86-21810e092527" width="50%">
</p>
---

### 🎥 Sample Video Predictions

[![Watch the demo]](https://github.com/user-attachments/assets/3cc99348-bd90-4a2d-8084-c2e9488e1a0a)

---

### 📊 Confusion Matrix & Metrics

<p align="center">
  <img src="Metrics Images/Models Metrics Radar.png" width="50%">
  <img src="Metrics Images/RandomForest Comparing Classes.png" width="50%">
  <img src="Metrics Images/DT Confusion Mtarix.png" width="50%">
  <img src="Metrics Images/SVM Performance Metrics.png" width="50%">
</p>

---

## 8️⃣ Installation

### Clone Repository

```bash
git clone https://github.com/Aliha7ish/Hand_Landmarks_Detection.git
cd Hands_Marks_Detection_main
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 9️⃣ Usage

### Live Camera

```bash
python main.py --mode camera
```

### Video File

```bash
python main.py --mode video --path path_to_video.mp4
```

### Image

```bash
python main.py --mode image --path path_to_image.jpg
```

Press **Q** to exit camera or video mode.

---

## 🔬 Research & Experiment Tracking

All experiments were tracked using **MLflow**, including:

- Model comparisons  
- Metric visualization  
- Confusion matrix analysis  
- Model registry lifecycle (Production / Archived)  

<p align="center">
  <img src="https://github.com/user-attachments/assets/917a6a43-1533-4d89-b45d-2c8cc7dbe834" width="60%">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/8a4e18a5-11ca-4666-b015-b2b2c8212e6a" width="60%">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/4f761c31-7e96-49cd-ad17-433df29bd3e9" width="60%">
</p>
