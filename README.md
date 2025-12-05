# Deep Learning–Driven Pneumonia Screening from X-Ray Images

### Performance Analysis of CNN (ResNet-50) and Vision Transformer (ViT-B/16)



## Overview
This project analyzes how deep learning models can detect pneumonia from chest X-ray images. We evaluated and compared two state-of-the-art architectures:

1.  **ResNet-50** (Convolutional Neural Network)
2.  **ViT-B/16** (Vision Transformer)

We performed experiments comparing imbalanced vs. balanced training, model fine-tuning, and model explainability using Grad-CAM and ViT attention maps.

## 📂 Dataset
The model was trained and evaluated on the **Kaggle Chest X-Ray Images (Pneumonia)** dataset.

🔗 **Dataset Link:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
| :--- | :---: | :---: | :---: |
| **Train** | 1,341 | 3,875 | 5,216 |
| **Validation** | 8 | 8 | 16 |
| **Test** | 234 | 390 | 624 |

**Issues Identified:** The dataset is highly imbalanced, causing initial models to incorrectly classify many NORMAL images as pneumonia.

## 🛠️ Preprocessing & Augmentation
To improve model generalization and handle class imbalance:

* **Resize:** 224×224
* **Augmentation:** Random horizontal flip, Random rotation (±10°)
* **Normalization:** Using ImageNet mean/std
* **Balancing:** Applied `WeightedRandomSampler` during training to oversample the minority class (Normal).

## 🧠 Models Implemented

### 1️. ResNet-50 (CNN)
* **Weights:** Pretrained on ImageNet
* **Configuration:** Last block unfrozen (`layer4`)
* **Head:** New fully connected head → 2 classes
* **Learning Rates:**
    * Backbone: `1e-5`
    * Classifier: `1e-4`

### 2️. Vision Transformer (ViT-B/16)
* **Weights:** Pretrained on ImageNet
* **Configuration:** Only the last encoder block unfrozen
* **Head:** New classification head (2-class output)
* **Learning Rates:**
    * Encoder block: `1e-5`
    * Head: `1e-4`

## 📊 Results

### Imbalanced Training Results
*Both models were initially biased toward predicting pneumonia due to the dataset imbalance.*

| Model | Accuracy | Normal Recall | Pneumonia Recall |
| :--- | :---: | :---: | :---: |
| **ResNet-50** | 0.80 | 0.48 | 0.99 |
| **ViT-B/16** | 0.81 | 0.51 | 0.99 |

### Balanced Training Results
*After applying `WeightedRandomSampler`:*

| Model | Accuracy | Normal Recall | Pneumonia Recall |
| :--- | :---: | :---: | :---: |
| **ResNet-50** | 0.85 | 0.61 | 0.99 |
| **ViT-B/16** | **0.91** | **0.79** | **0.98** |

**Observation:** ViT shows far more balanced results, specifically improving performance on NORMAL cases while maintaining high sensitivity for Pneumonia.

### Final Fine-Tuned ViT Results (Best Model)
**Overall Accuracy:** 0.92

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.96 | 0.82 | 0.89 |
| **Pneumonia** | 0.90 | 0.98 | 0.94 |

## Explainability

### Grad-CAM (ResNet-50)
* Heatmaps focus on small, localized regions.
* **Issue:** Often misinterprets normal shadows as pneumonia.
* *Example:* In a TRUE-NORMAL image, ResNet activated a left-lung shadow and predicted pneumonia.

### ViT Attention Maps
* Uses distributed patch-based attention across lung fields.
* Captures more global and stable features.
* **Result:** Correctly identified the same challenging case as NORMAL.

## 📉 Limitations
* **Validation Set:** Very small (only 16 images).
* **Demographics:** Dataset is pediatric; results may not generalize to adults.
* **Output:** Only binary classification (Pneumonia vs Normal).
* **Calibration:** No probability calibration implemented.
* **Tuning:** Limited hyperparameter tuning due to compute constraints.

## 🚀 Future Work
* Implement multi-class pneumonia classification (Viral vs Bacterial).
* Extend architecture to 3D CT Scan models.
* Deploy as a mobile/web screening tool.
* Improve explainability (e.g., using Attention Rollout).

## 👥 Authors & Contributions
* **Venkata Jyothi Priya Mulaka -** Technical Implementation, code & Presentation.
* **Srikar Gowrishetty -** Report & Presentation.
* **Eli Antoine -** Report & Presentation.
