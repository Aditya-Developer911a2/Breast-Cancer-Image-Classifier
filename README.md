# Breast Cancer Histopathology Classification (BreaKHis)

A reproducibility-focused comparison of transfer learning strategies for breast
cancer histopathological image classification, with a deployed demo app.

**[Live Demo](#)** *(add your Streamlit Cloud URL here once deployed)*

> ⚠️ **Research and educational project only.** Not a diagnostic tool, not
> clinically validated. See [Limitations](#limitations) before drawing any
> conclusions from it.

---

## Overview

This project evaluates a Baseline CNN, ResNet50, and EfficientNetB0 on the
[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
dataset, under a strict **patient-level** train/validation/test split to
prevent data leakage. The two pretrained architectures were adapted using a
**two-phase fine-tuning strategy**: a frozen-backbone phase to train a new
classification head, followed by partial backbone unfreezing (top 40% of
layers) with BatchNormalization layers kept frozen throughout.

The full experimental protocol was repeated across **five random seeds** to
measure reproducibility — a step most published work in this space skips,
reporting only a single run.

## Dataset

| | |
|---|---|
| Source | [BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) |
| Images | 7,909 |
| Patients | 81 |
| Classes | Benign (2,480) / Malignant (5,429) |
| Magnifications | 40X, 100X, 200X, 400X |
| Split | 70% / 15% / 15% (patient-level, leakage-verified) |

## Methodology

1. **Baseline CNN** — trained from scratch, no pretrained weights (reference point).
2. **ResNet50 / EfficientNetB0** — two-phase fine-tuning:
   - Phase 1: backbone frozen, train classification head only.
   - Phase 2: top 40% of backbone unfrozen, BatchNorm layers stay frozen,
     learning rate reduced 100x.
3. **Multi-seed evaluation** — identical pipeline repeated across seeds
   `42, 123, 456, 789, 2026`, reporting mean ± standard deviation.
4. Evaluated at both **image level** and **patient level** (averaging
   per-image probability within each patient), plus test-time augmentation
   and per-magnification breakdown.

## Results

**Image-level (mean ± SD, 5 seeds)**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline CNN | 81.04% ± 7.57 | 93.23% ± 6.74 | 80.13% ± 16.14 | 84.93% ± 8.18 | 94.43% ± 1.22 |
| ResNet50 | 89.22% ± 1.07 | 97.05% ± 0.94 | 87.34% ± 0.81 | 91.94% ± 0.80 | 97.38% ± 0.88 |
| EfficientNetB0 | 88.24% ± 2.44 | 98.73% ± 0.26 | 84.37% ± 3.44 | 90.95% ± 2.04 | 96.96% ± 0.56 |

**Patient-level (mean ± SD, 5 seeds)**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline CNN | 84.62% ± 13.32 | 98.00% ± 4.47 | 80.00% ± 21.37 | 86.40% ± 14.21 | 98.33% ± 1.52 |
| ResNet50 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| EfficientNetB0 | 92.31% ± 0.00 | 100.00% ± 0.00 | 88.89% ± 0.00 | 94.12% ± 0.00 | 100.00% ± 0.00 |

**Key finding:** the Baseline CNN's patient-level recall varied by over 20
points across seeds (80.00% ± 21.37) — training from scratch on this
dataset size is meaningfully unreliable. Both transfer-learning models
showed image-level standard deviations under 2.5 points on every metric,
and zero patient-level variance across all five seeds. Transfer learning's
value here isn't just accuracy — it's *reliability of the training process
itself*.

*(Add your ROC curve, PR curve, and magnification-wise figures here —
`docs/roc_curve.png` etc.)*

## Demo App

The deployed app (`app.py`) loads a single trained checkpoint
(EfficientNetB0, seed 42 by default — see `src/predict.py` to change) and
serves predictions on uploaded histopathology images.

### Running locally

```bash
git clone https://github.com/<your-username>/breakhis-classifier.git
cd breakhis-classifier
pip install -r requirements.txt
```

Place your trained model file at `model/efficientnetb0_seed42.keras`
(or update `LOCAL_MODEL_PATH` in `src/predict.py`), then:

```bash
streamlit run app.py
```

### Getting the model file onto the machine

The trained model isn't small enough to comfortably commit directly to
GitHub. Two options:

**Option A — Git LFS** (simplest if you're comfortable with it):
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes model/efficientnetb0_seed42.keras
git commit -m "Add model via Git LFS"
```

**Option B — Hugging Face Hub** (recommended for deployment):
1. Create a model repo at [huggingface.co/new](https://huggingface.co/new).
2. Upload your `.keras` file there.
3. Set the `HF_MODEL_REPO` environment variable (locally) or Streamlit
   secret (when deployed) to your repo ID, e.g. `yourname/breakhis-effnetb0`.
4. Remove the model file from `model/` and from git tracking — `src/predict.py`
   will download it automatically at startup instead.

### Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub account, and select this repo, with `app.py` as the entry point.
3. If using the Hugging Face Hub option, add `HF_MODEL_REPO` under
   **Settings → Secrets** in the Streamlit Cloud dashboard.
4. Deploy. First load will be slow (downloading TensorFlow + model);
   subsequent loads are cached.

## Repository Structure

```
breakhis-classifier/
├── app.py                  # Streamlit app entry point
├── src/
│   ├── preprocess.py        # Image preprocessing (must match training exactly)
│   └── predict.py           # Model loading + inference
├── model/                   # Trained model file goes here (see above)
├── notebooks/                # Training notebooks (add your own)
├── requirements.txt
└── README.md
```

## Limitations

- **Not clinically validated.** Do not use for actual diagnosis.
- **Small patient-level test set** (~12 patients) — strong results, but
  on a small sample; treat perfect scores with appropriate skepticism
  about generalization.
- **No frozen-backbone-only control** was run, so the specific
  contribution of partial unfreezing (vs. transfer learning generally)
  isn't isolated in this study.
- **Single-dataset training** — performance on histopathology images from
  different labs, stains, or scanners is untested.
- **ROC/PR curves** (if included above) are drawn from a single
  representative seed, not averaged across all five.

## Future Work

- Frozen-backbone control condition through the same multi-seed pipeline.
- Patient-level k-fold cross-validation.
- Magnification-specific fine-tuning.
- Investigate why test-time augmentation's effect was architecture-dependent.

## Acknowledgments

Built as part of a research internship at [Institution]. Dataset: BreaKHis
[Spanhol et al., 2016]. Architectures: ResNet50 [He et al., 2016],
EfficientNetB0 [Tan & Le, 2019].

## License

*(Add a license — MIT is a reasonable default for a portfolio project;
note that the BreaKHis dataset itself has its own usage terms, so check
those separately before redistributing any data.)*
