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

    42, 123, 456, 789, 2026

Results are reported as **mean ± standard deviation** across the five seeds.

---

## Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

Evaluation was performed at both:

- **Image level**
- **Patient level**

Additional experiments included:

- Test-time augmentation
- Performance across 40X, 100X, 200X and 400X magnifications

---

## Results

### Image-Level Performance

Mean ± standard deviation across five seeds.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 81.04% ± 7.57 | 93.23% ± 6.74 | 80.13% ± 16.14 | 84.93% ± 8.18 | 94.43% ± 1.22 |
| ResNet50 | **89.22% ± 1.07** | 97.05% ± 0.94 | 87.34% ± 0.81 | **91.94% ± 0.80** | **97.38% ± 0.88** |
| EfficientNetB0 | 88.24% ± 2.44 | **98.73% ± 0.26** | 84.37% ± 3.44 | 90.95% ± 2.04 | 96.96% ± 0.56 |

### Patient-Level Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 84.62% ± 13.32 | 98.00% ± 4.47 | 80.00% ± 21.37 | 86.40% ± 14.21 | 98.33% ± 1.52 |
| ResNet50 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| EfficientNetB0 | 92.31% ± 0.00 | 100.00% ± 0.00 | 88.89% ± 0.00 | 94.12% ± 0.00 | 100.00% ± 0.00 |

### Key Finding

Transfer learning substantially outperformed the baseline CNN and showed much
lower seed-to-seed variation.

The baseline CNN's patient-level recall varied by more than 20 percentage
points across five seeds, while both transfer-learning models showed much
greater stability.

---

## Repository Structure

    breakhis-classifier/
    │
    ├── model/
    │   └── trained model files
    │
    ├── src/
    │   ├── __init__.py
    │   ├── predict.py
    │   └── preprocess.py
    │
    ├── .gitignore
    ├── README.md
    └── requirements.txt

### Source Files

- `preprocess.py` — image preprocessing and input preparation.
- `predict.py` — model loading and prediction pipeline.
- `model/` — trained model files used for inference.

The repository contains the implementation code, trained model files and
supporting configuration used during the experiments.

---

## Limitations

- This is a research and educational project and is **not clinically validated**.
- The patient-level test set contains approximately 12 patients, so patient-level
  performance should be interpreted cautiously.
- A frozen-backbone-only control was not included, so the individual contribution
  of partial unfreezing cannot be isolated from transfer learning as a whole.
- The experiments use a single dataset; generalization to images from different
  laboratories, scanners or staining protocols was not evaluated.
- ROC and precision-recall curves were generated from a representative seed
  rather than averaged across all five seeds.

---

## Future Work

- Add a frozen-backbone control condition.
- Perform patient-level k-fold cross-validation.
- Investigate magnification-specific fine-tuning.
- Analyze the architecture-dependent effect of test-time augmentation.
- Average ROC and precision-recall curves across all random seeds.

---

## Acknowledgments

This project was conducted as part of a research internship at the
**Department of Computer Science and Engineering, Motilal Nehru National
Institute of Technology Allahabad**, under the guidance of **Dr. Ranvijay**.

Dataset: [BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

Architectures:

- ResNet50
- EfficientNetB0
