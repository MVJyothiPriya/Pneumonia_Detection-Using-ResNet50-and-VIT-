# Deep Learning–Driven Pneumonia Screening from X-Ray Images

### Performance Analysis of CNN (ResNet-50) and Vision Transformer (ViT-B/16)

**Authors:** Venkata Jyothi Priya Mulaka, Srikar Gowrishetty, Eli Antoine
**Affiliation:** University of Florida

---

## 📌 Overview
This project analyzes how deep learning models can detect pneumonia from chest X-ray images. We evaluated and compared two state-of-the-art architectures:

1.  **ResNet-50** (Convolutional Neural Network)
2.  **ViT-B/16** (Vision Transformer)

We performed experiments comparing imbalanced vs. balanced training, model fine-tuning, and model explainability using Grad-CAM and ViT attention maps.

> **Note:** Full project details are available in our submitted report and presentation.

## 📂 Dataset
**Source:** Kaggle Chest X-Ray Pneumonia Dataset

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
* **Balancing:** Applied `WeightedRandomSampler` during training.

## 🧠 Models Implemented

### 1️⃣ ResNet-50 (CNN)
* **Weights:** Pretrained on ImageNet
* **Configuration:** Last block unfrozen (`layer4`)
* **Head:** New fully connected head → 2 classes
* **Learning Rates:**
    * Backbone: `1e-5`
    * Classifier: `1e-4`

### 2️⃣ Vision Transformer (ViT-B/16)
* **Weights:** Pretrained on ImageNet
* **Configuration:** Only the last encoder block unfrozen
* **Head:** New classification head (2-class output)
* **Learning Rates:**
    * Encoder block: `1e-5`
    * Head: `1e-4`

## 📊 Results

### 🚫 Imbalanced Training Results
*Both models were initially biased toward predicting pneumonia.*

| Model | Accuracy | Normal Recall | Pneumonia Recall |
| :--- | :---: | :---: | :---: |
| **ResNet-50** | 0.80 | 0.48 | 0.99 |
| **ViT-B/16** | 0.81 | 0.51 | 0.99
