# ✋ Hand Gesture Classification Using Landmark Features

---

## 1️⃣ Objective

This research branch evaluates multiple classical machine learning models for hand gesture classification using landmark coordinates as input features.

The purpose of this study is to:

- Compare model performance across training, development, and test datasets  
- Analyze overfitting and generalization behavior  
- Investigate class-level confusion patterns  
- Study ambiguity between visually similar gestures ("peace" vs "two_up")  
- Select the most suitable model for deployment  

---

## 2️⃣ Experimental Setup

### 📊 Dataset

- 20,540 training samples  
- Separate development and test sets  
- Input: Hand landmark coordinates  
- Output: Gesture class label  

<p align="center">
  <img src="Metrics Images/datasets_distribution.png" width="50%">
</p>

### 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  

All experiments were tracked using **MLflow**, and performance visualizations were generated for deeper analysis.

---

## 3️⃣ Models Evaluated

- Decision Tree  
- Random Forest  
- Support Vector Machine (RBF Kernel)  

---

## 4️⃣ Quantitative Performance Comparison

Below are the performance summaries of the three models.

<p align="center">
  <img src="Metrics Images/DT Performance Metrics.png" width="33%">
  <img src="Metrics Images/RandomForest Performance Metrics.png" width="33%">
  <img src="Metrics Images/SVM Performance Metrics.png" width="33%">
</p>

### 🔎 Observations

#### Decision Tree
- 100% training accuracy  
- ~6% drop on test set  
- Clear sign of overfitting  

#### Random Forest
- 100% training accuracy  
- ~97% on development and test sets  
- Only ~2.5% drop  
- Strong generalization  

#### SVM (RBF)
- Lower training performance than Random Forest  
- Test performance slightly higher than dev  
- No evidence of overfitting  
- Most stable generalization pattern  

---

## 5️⃣ Confusion Matrix Analysis

<p align="center">
  <img src="Metrics Images/DT Confusion Mtarix.png" width="50%">
</p>
<p align="center">
  <img src="Metrics Images/RandomForest Confusion Matrix.png" width="50%">
</p>
<p align="center">
  <img src="Metrics Images/SVM Confusion Matrix.png" width="50%">
</p>

### 🌳 Decision Tree

The Decision Tree shows high training performance but struggles in unseen environments due to overly specific decision boundaries. Its deep structure likely memorized precise landmark configurations rather than learning general geometric relationships.

---

### 🌲 Random Forest

The Random Forest correctly classifies the vast majority of gestures across rotations and finger spread variations.

**Notable findings:**

- Only 5 misclassifications in development set  
- Slight difficulty with inverted "peace" gesture (~93.5%)  
- Handles natural human variation effectively  

---

### 🧠 Support Vector Machine (RBF)

The SVM demonstrates stable generalization but shows increased confusion between visually similar gestures.

**Observed behavior:**

- 27 misclassifications between "peace" and "two_up"  
- Struggles more with tilted hands (~45° rotations)  
- RBF kernel classifies based on proximity to support vectors  
- Narrow finger spread in "peace" sometimes shifts classification toward "two_up"  

---

## 6️⃣ Correct Predictions Visualization

Below are examples of correctly classified gestures per model.

<p align="center">
  <img src="Metrics Images/DT True Gestures.png" width="33%">
  <img src="Metrics Images/RandomForest True Predictions.png" width="33%">
  <img src="Metrics Images/SVM True Predictions.png" width="33%">
</p>

### 💡 Insight

Random Forest remains robust across:

- Rotation variations  
- Slight finger spread differences  
- Minor landmark noise  

SVM shows sensitivity to global hand orientation.

Decision Tree appears confident but less adaptable to variations.

---

## 7️⃣ Two-Class Deep Dive: "peace" vs "two_up"

These two gestures are geometrically similar and represent the most challenging classification boundary.

Below are TP / TN / FP / FN comparisons for each model.

- Decision Tree  
- Random Forest  
- SVM  

<p align="center">
  <img src="Metrics Images/DT Comparing Classes.png" width="33%">
  <img src="Metrics Images/RandomForest Comparing Classes.png" width="33%">
  <img src="Metrics Images/SVM Comparing Classes.png" width="33%">
</p>

### Findings

- Random Forest handles finger spread ambiguity better.  
- SVM shows symmetric FP/FN behavior (no class bias).  
- Decision Tree produces sharper but less flexible decision regions.  

---

## 8️⃣ Radar Plot – Multi-Metric Comparison

The radar plot summarizes Accuracy, Precision, Recall, and F1-score across models.
<p align="center">
  <img src="Metrics Images/Models Metrics Radar.png" width="50%">
</p>


### Interpretation

- Random Forest dominates across all metrics.  
- SVM is stable but consistently 4–5% lower.  
- Decision Tree shows strong training metrics but weaker generalization.  

---

## 9️⃣ Model Selection Decision

# 🏆 Selected Model: Random Forest

### Why?

- Highest overall performance (~97.4%)  
- Minimal overfitting (2.5% gap)  
- Handles geometric ambiguity better than SVM  
- More robust to hand rotation and finger variation  
- Efficient on CPU  
- Better suited for real-world "in-the-wild" scenarios  

---

## 🔟 Final Thoughts

- Decision Tree is fast but overfits.  
- SVM generalizes well but is sensitive to hand orientation.  
- Random Forest achieves the best trade-off between:
  - Accuracy  
  - Stability  
  - Robustness  
  - Computational efficiency  

For deployment and live inference scenarios, Random Forest is the most reliable candidate.

However, SVM remains a strong secondary option if Random Forest underperforms in streaming or unseen environments.


