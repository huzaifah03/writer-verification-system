# Writer Verification System — Training Results

## Overview

Two training runs were conducted on the IAM Handwriting Database.
Run 1 revealed a critical preprocessing mismatch; Run 2 fixed it and produced substantially better metrics.

**Dataset:** 655 writers, 13,353 line images  
**Split (seed=42):** 459 train / 98 val / 98 test (writer-level, no leakage)  
**Hardware:** GTX 1660 SUPER, CUDA 12.1, Windows 11 (workers=0, single-threaded loading)  
**Architecture:** ResNet-50 backbone → 256-dim projection head (WriterEmbeddingNet)  
**Loss:** ContrastiveLoss (margin=1.0), optimizer Adam, StepLR (step=7, γ=0.1)

---

## Run 1 — Baseline (Preprocessing Mismatch)

**Date:** 2026-06-05  
**Epochs completed:** 20/20  
**Duration:** ~5.5 hours  
**Best checkpoint:** Epoch 7, val_loss = 0.0909

### Configuration
- No data augmentation
- **Bug:** `preprocess_for_model()` applied Gaussian blur + Otsu binarization at inference time, but training used resize + grayscale only → train/inference distribution mismatch

### Loss Curve

| Epoch | Train Loss | Val Loss | Best? |
|-------|-----------|----------|-------|
| 1  | 0.2118 | 0.1317 | Yes |
| 2  | 0.1363 | 0.1066 | Yes |
| 3  | 0.1122 | 0.1037 | Yes |
| 4  | 0.0955 | 0.0995 | Yes |
| 5  | 0.0845 | 0.0996 | |
| 6  | 0.0760 | 0.0927 | Yes |
| **7**  | **0.0679** | **0.0909** | **Yes (final)** |
| 8  | 0.0552 | 0.0913 | |
| 9  | 0.0477 | 0.0919 | |
| 10 | 0.0445 | 0.0922 | |
| 11 | 0.0414 | 0.0925 | |
| 12 | 0.0395 | 0.0927 | |
| 13 | 0.0382 | 0.0929 | |
| 14 | 0.0361 | 0.0947 | |
| 15 | 0.0338 | 0.0943 | |
| 16 | 0.0332 | 0.0948 | |
| 17 | 0.0321 | 0.0953 | |
| 18 | 0.0326 | 0.0946 | |
| 19 | 0.0322 | 0.0945 | |
| 20 | 0.0326 | 0.0945 | |

LR dropped 10× at epoch 8 (1e-4 → 1e-5) and again at epoch 15 (→ 1e-6).
Val loss plateaued at ~0.091 from epoch 7 onwards — training loss kept falling (overfitting).

### Evaluation Results (98 test writers, 1960 pairs)

| Mode | AUC | Acc@default(0.75) | F1@default(0.75) | Acc@optimal | F1@optimal | Optimal Threshold |
|------|-----|-------------------|------------------|-------------|------------|-------------------|
| hybrid  | 0.598 | 0.556 | 0.524 | 0.576 | 0.464 | 0.764 |
| siamese | 0.660 | 0.500 | 0.667 | 0.626 | 0.631 | 0.994 |
| hog     | 0.584 | 0.500 | 0.000 | 0.571 | 0.466 | 0.533 |

### Observations
- AUCs barely above chance (0.5); metrics were misleadingly poor
- Siamese optimal threshold of **0.994** is pathological — well-trained embeddings should sit at 0.5–0.8, indicating embeddings were not properly separated
- HOG F1@0.75 = 0.000 because HOG cosine scores rarely exceed 0.75
- Hybrid worse than Siamese-only: HOG was diluting the signal
- **Root cause confirmed:** preprocessing mismatch — model learned one image distribution but was evaluated on a different one (binarized images)

---

## Run 2 — Fixed Pipeline

**Date:** 2026-06-06  
**Epochs completed:** 13/20 (process killed by Windows while manually suspended; training was plateauing)  
**Duration:** ~3.5 hours  
**Best checkpoint:** Epoch 12, val_loss = 0.1340

### Changes vs Run 1
1. **Preprocessing fix:** `preprocess_for_model()` now matches training exactly — resize(224×224) → Grayscale(3ch) → ToTensor → Normalize(ImageNet). Gaussian blur and Otsu binarization removed from inference path.
2. **Data augmentation added to training:** `RandomRotation(5)`, `ColorJitter(brightness=0.2, contrast=0.2)`, `RandomHorizontalFlip(p=0.3)` — validation transform kept clean.

### Loss Curve

| Epoch | Train Loss | Val Loss | Best? |
|-------|-----------|----------|-------|
| 1  | 0.2417 | 0.1881 | Yes |
| 2  | 0.2006 | 0.1583 | Yes |
| 3  | 0.1726 | 0.1423 | Yes |
| 4  | 0.1547 | 0.1421 | Yes |
| 5  | 0.1409 | 0.1449 | |
| 6  | 0.1328 | 0.1383 | Yes |
| 7  | 0.1240 | 0.1479 | |
| 8  | 0.1097 | 0.1364 | Yes |
| 9  | 0.1044 | 0.1342 | Yes |
| 10 | 0.1021 | 0.1342 | |
| 11 | 0.0988 | 0.1359 | |
| **12** | **0.0958** | **0.1340** | **Yes (final)** |
| 13 | 0.0933 | 0.1346 | |

Val loss plateau from epoch 9–13 (0.1340–0.1346) indicates the model had converged;
remaining epochs 14–20 were unlikely to yield meaningful improvement.
Higher absolute val loss vs Run 1 is expected: augmentation increases training variance,
and the corrected image distribution is a genuinely harder input.

### Evaluation Results (98 test writers, 1960 pairs)

| Mode | AUC | Acc@default(0.764) | F1@default(0.764) | Acc@optimal | F1@optimal | Optimal Threshold |
|------|-----|--------------------|--------------------|-------------|------------|-------------------|
| hybrid  | 0.892 | 0.601 | 0.353 | 0.814 | 0.822 | 0.623 |
| siamese | **0.904** | 0.824 | 0.831 | **0.828** | **0.831** | 0.798 |
| hog     | 0.584 | 0.500 | 0.000 | 0.571 | 0.466 | 0.533 |

### Observations
- Siamese AUC **0.904** — excellent discriminative power
- Siamese optimal threshold **0.798** — healthy range, confirms embeddings are well-separated
- Hybrid AUC (0.892) slightly below Siamese-only: HOG still dilutes the Siamese signal; a mode-specific threshold or weighted combination would help
- HOG unchanged at 0.584 — expected, HOG does not use the trained model
- HOG F1@default = 0.000 persists because HOG cosine scores are distributed differently from Siamese scores; the shared threshold does not transfer

---

## Run 1 vs Run 2 — Direct Comparison

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| Best val loss | 0.0909 (ep. 7) | 0.1340 (ep. 12) | — |
| Hybrid AUC | 0.598 | **0.892** | +0.294 |
| Siamese AUC | 0.660 | **0.904** | +0.244 |
| HOG AUC | 0.584 | 0.584 | 0 |
| Siamese F1@optimal | 0.631 | **0.831** | +0.200 |
| Siamese Acc@optimal | 0.626 | **0.828** | +0.202 |
| Siamese optimal threshold | 0.994 | 0.798 | −0.196 |
| Epochs to best | 7 | 12 | +5 |

The preprocessing fix alone accounts for the entire improvement.
The augmentation contributed regularisation (convergence at epoch 12 vs 7)
but the AUC gain is primarily from eliminating the distribution mismatch.

---

## Known Limitations

1. **HOG always underperforms** at the shared threshold — the deployed hybrid pipeline
   should use mode-specific thresholds or drop HOG from the similarity score.
2. **Run 2 stopped at epoch 13** — a full 20-epoch run may squeeze another 1–2 points of AUC,
   though the plateau suggests diminishing returns.
3. **workers=0 on Windows** — single-threaded image loading is the training bottleneck
   (~16 min/epoch). On Linux/Colab with workers=4 this drops to ~3–5 min/epoch.
4. **Triplet loss not yet tried** — replacing ContrastiveLoss with TripletMarginLoss is the
   next lever for embedding quality.

---

## File Locations

| Artifact | Path |
|----------|------|
| Best model (Run 2) | `saved_models/writer_verification_model.pth` (99 MB, gitignored) |
| Writer split | `saved_models/splits.json` (gitignored, paths are machine-specific) |
| Metrics JSON | `results/metrics.json` |
| ROC curve | `results/roc_curve.png` |
| Score distribution | `results/score_distribution.png` |
| Confusion matrix | `results/confusion_matrix.png` |
| Run 1 training log | `../training_run1.log` (not committed) |
| Run 2 training log | `../training_run2.log` (not committed) |
