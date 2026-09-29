# Breast Cancer Histopathology Classification (BreaKHis)

A reproducibility-focused study comparing transfer learning strategies for
breast cancer histopathological image classification using the BreaKHis dataset.

> **Research and educational project only.** This project is not a diagnostic
> tool and has not been clinically validated.

---

## Overview

This project evaluates three CNN-based approaches for breast cancer
histopathological image classification:

- Baseline CNN trained from scratch
- ResNet50 with ImageNet-pretrained weights
- EfficientNetB0 with ImageNet-pretrained weights

The transfer-learning models use a two-phase fine-tuning strategy:

1. Freeze the pretrained backbone and train a new classification head.
2. Unfreeze the top 40% of backbone layers while keeping BatchNormalization
   layers frozen and reducing the learning rate.

The complete experimental pipeline was repeated across **five random seeds**
to study reproducibility and seed-to-seed variation.

---

## Dataset

The project uses the **BreaKHis** breast cancer histopathological image dataset.

| Property | Details |
|---|---|
| Images | 7,909 |
| Patients | 81 |
| Classes | Benign: 2,480 / Malignant: 5,429 |
| Magnifications | 40X, 100X, 200X, 400X |
| Train / Validation / Test | 70% / 15% / 15% |
| Split strategy | Patient-level, leakage-verified |

Dataset: [BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

---

## Methodology

### 1. Baseline CNN

A CNN was trained from scratch as a non-transfer-learning reference model.

### 2. Transfer Learning

Two ImageNet-pretrained architectures were evaluated:

- **ResNet50**
- **EfficientNetB0**

Both models used the same two-phase fine-tuning strategy:

- **Phase 1:** Freeze the pretrained backbone and train the classification head.
- **Phase 2:** Unfreeze the top 40% of backbone layers.
- BatchNormalization layers remained frozen during fine-tuning.
- The learning rate was reduced from `1e-3` to `1e-5`.

### 3. Data Preprocessing

- Patient-level stratified train/validation/test split
- Explicit verification of zero patient overlap between splits
- Images resized to `224 × 224`
- ImageNet preprocessing for ResNet50 and EfficientNetB0
- Training-time augmentation
- Inverse-frequency class weighting to handle class imbalance

### 4. Reproducibility

The complete training and evaluation pipeline was repeated using five random
seeds:

```text
42, 123, 456, 789, 2026
