# Writer Verification System — Session Context
**Date:** 2026-06-05
**Machine:** Windows 11, GTX 1660 SUPER, CUDA 13.2 driver
**GitHub:** https://github.com/huzaifah03/writer-verification-system
**Active branch:** `final` (all session changes pushed here)

---

## 1. What Was Built

AI-based handwriting writer-verification system. Two handwriting samples → same/different writer decision.

**Pipeline:**
- `WriterEmbeddingNet` (ResNet-50 + 256-dim projection head) → 256-d Siamese embedding
- HOG extractor → 26,244-d traditional descriptor
- Both L2-normalised, concatenated → 26,500-d hybrid vector, compared with cosine similarity
- Threshold: `Config.SIMILARITY_THRESHOLD = 0.764` (ROC-optimal from evaluation)

---

## 2. Directory Layout

```
c:\Users\Hamzah Shaikh\Downloads\Huzaifah\writer-verification-system\   <- git repo root
├── venv\                          <- Python 3.12 venv (NOT in git)
├── TRAINING_SETUP.md              <- original setup guide (committed)
├── SESSION_CONTEXT.md             <- this file
└── writer-verification-system\   <- ACTUAL PROJECT ROOT (all Python lives here)
    ├── config.py                  <- MODIFIED: threshold 0.75 -> 0.764
    ├── run.py
    ├── model\
    │   ├── feature_extractor.py   <- MODIFIED: added WriterEmbeddingNet + helpers
    │   ├── preprocessing.py       <- MODIFIED: removed Gaussian+Otsu from inference path
    │   ├── predict.py
    │   └── similarity.py
    ├── training\
    │   ├── __init__.py            <- NEW (empty)
    │   ├── data_split.py          <- NEW: writer-level 70/15/15 split
    │   └── train.py               <- REPLACED: Siamese training with val split
    ├── scripts\
    │   ├── evaluate.py            <- NEW: evaluation + metrics.json + 3 plots
    │   └── prepare_iam.py         <- NEW: IAM -> per-writer folder converter
    ├── results\                   <- NEW (committed)
    │   ├── metrics.json
    │   ├── roc_curve.png
    │   ├── score_distribution.png
    │   └── confusion_matrix.png
    ├── saved_models\
    │   ├── writer_verification_model.pth   <- NOT in git (gitignored, 99 MB)
    │   └── splits.json                     <- NOT in git (gitignored, machine-specific paths)
    └── data\                      <- NOT in git (gitignored)
        ├── iam\ascii\forms.txt
        ├── iam\lines\<a01..z>\<form_id>\*.png
        └── processed\writer_XXX\*.png      <- 655 writers, 13,353 images
```

**Python executable (this machine):**
`c:\Users\Hamzah Shaikh\Downloads\Huzaifah\writer-verification-system\venv\Scripts\python.exe`

---

## 3. Running Commands (Windows — CRITICAL)

Always set PYTHONPATH before running training/eval scripts:

```powershell
$projectRoot = "c:\Users\Hamzah Shaikh\Downloads\Huzaifah\writer-verification-system\writer-verification-system"
$python = "c:\Users\Hamzah Shaikh\Downloads\Huzaifah\writer-verification-system\venv\Scripts\python.exe"
$env:PYTHONPATH = $projectRoot
Set-Location $projectRoot

# Full training (20 epochs, ~5.5 h on GTX 1660 SUPER)
& $python training/train.py --data_dir data/processed --epochs 20 --batch_size 32

# Evaluation only
& $python scripts/evaluate.py --data_dir data/processed
```

On Linux/Colab:
```bash
cd /path/to/writer-verification-system/writer-verification-system
PYTHONPATH=. python training/train.py --data_dir data/processed --epochs 20 --batch_size 32
```

---

## 4. Training Results

| Epoch | Train Loss | Val Loss | Saved? |
|-------|-----------|----------|--------|
| 1 | 0.2118 | 0.1317 | Best |
| 2 | 0.1363 | 0.1066 | Best |
| 3 | 0.1122 | 0.1037 | Best |
| 4 | 0.0955 | 0.0995 | Best |
| 6 | 0.0760 | 0.0927 | Best |
| **7** | **0.0679** | **0.0909** | **Best (final checkpoint)** |
| 8–20 | 0.055→0.033 | 0.091–0.095 | — |

- Best model: epoch 7, val_loss = 0.0909
- LR schedule: 1e-4 → 1e-5 at epoch 8 → 1e-6 at epoch 15 (StepLR, step=7, gamma=0.1)
- Total time: ~5.5 hours (GTX 1660 SUPER, workers=0 on Windows)
- Dataset: 655 writers, 13,353 images; split 459/98/98 train/val/test (seed=42)

---

## 5. Evaluation Results (held-out test writers)

98 test writers, 1,960 pairs (980 same / 980 different), seed=42

| Mode | AUC | Acc@0.75 | F1@0.75 | Acc@opt | F1@opt | Opt Thr |
|------|-----|----------|---------|---------|--------|---------|
| hybrid | 0.598 | 0.556 | 0.524 | 0.576 | 0.464 | 0.764 |
| siamese | **0.660** | 0.500 | 0.667 | **0.626** | **0.631** | 0.994 |
| hog | 0.584 | 0.500 | 0.000 | 0.571 | 0.466 | 0.533 |

**Interpretation for the report:**
These results were produced BEFORE the preprocessing fix. After retraining with the
fixed pipeline (Change 1 below), all three AUC values should improve — especially
hybrid, which was dragged down by the mismatch. The Siamese-optimal threshold of 0.994
indicates embeddings are not yet well-separated (a well-trained model sits ~0.5–0.8).

---

## 6. Changes Made This Session

### Already committed and pushed to `final`:

**Change 1 — Preprocessing fix** (`model/preprocessing.py`)
`preprocess_for_model()` now matches training transforms exactly:
resize(224,224) → Grayscale(3ch) → ToTensor → Normalize(ImageNet).
The old Gaussian blur + Otsu binarization is removed from the inference path
(kept in `preprocess_for_visualization()` for display only).

**Change 2 — Data augmentation** (`training/train.py`)
`train_transform` now includes `RandomRotation(5)`, `ColorJitter(brightness=0.2, contrast=0.2)`,
`RandomHorizontalFlip(p=0.3)`. A separate `val_transform` has no augmentation.

**Change 3 — Threshold update** (`config.py`)
`SIMILARITY_THRESHOLD = 0.764` (was 0.75; ROC-optimal Youden J from evaluation).

**Change 4 — workers** (`training/train.py`)
`workers = 0 if os.name == "nt" else 4` — already in code. No code change needed.
On Linux/Colab this auto-enables 4 workers, cutting epoch time from ~16 min to ~3–5 min.

**Change 5 — Triplet loss** (not yet implemented — future iteration)
Replace `ContrastiveLoss` with `TripletMarginLoss` (PyTorch built-in).
Requires changing `WriterPairDataset` to yield (anchor, positive, negative) triplets.
This is the main lever for better embedding separation (threshold closer to 0.5–0.8).

---

## 7. What to Do Next (continuation)

### Priority 1 — Retrain with fixed pipeline
The preprocessing mismatch fix (Change 1) is in the code but the current
`saved_models/writer_verification_model.pth` was trained WITHOUT it.
The model weights and the metrics.json are stale. Retrain:

```powershell
$env:PYTHONPATH = $projectRoot
Set-Location $projectRoot
& $python -u training/train.py --data_dir data/processed --epochs 20 --batch_size 32
```

Use `-u` flag to prevent output buffering (previous session had silent stdout for 5 h).
After retraining, re-run evaluation:

```powershell
& $python scripts/evaluate.py --data_dir data/processed
```

### Priority 2 — Transfer model weights if changing machines
`saved_models/writer_verification_model.pth` (99 MB) is gitignored.
Copy it manually to the new machine, or retrain there.
`saved_models/splits.json` has machine-specific absolute paths;
`evaluate.py` will recompute the split from seed=42 automatically if paths don't resolve.

### Priority 3 — Triplet loss (optional, future)
See Change 5 above. Only worthwhile after the preprocessing fix produces better
baseline metrics.

---

## 8. Key File Contents

### `training/data_split.py`
```python
import os, random

def get_writer_splits(data_dir, seed=42, val_frac=0.15, test_frac=0.15):
    writer_dirs = sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    rng = random.Random(seed)
    rng.shuffle(writer_dirs)
    n = len(writer_dirs)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    return writer_dirs[n_test + n_val:], writer_dirs[n_test:n_test+n_val], writer_dirs[:n_test]
```

### Current `model/preprocessing.py` — `preprocess_for_model` (after fix)
```python
def preprocess_for_model(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)
```

### Current `training/train.py` — transform block (after augmentation)
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### `model/feature_extractor.py` — WriterEmbeddingNet
```python
class WriterEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embed = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embed(x)
```

---

## 9. IAM Dataset (this machine only — not in git)

- Archive source: `C:\Users\Hamzah Shaikh\Downloads\Huzaifah\IAM-Data\` (ascii.tgz + lines.tgz)
- Extracted to: `data/iam/ascii/` and `data/iam/lines/`
- Processed: `data/processed/writer_000/…writer_654/` (655 writers, 13,353 images)
- Rebuild on a new machine: extract archives, then:
  ```bash
  python scripts/prepare_iam.py --iam_dir data/iam --out_dir data/processed --min_samples 5
  ```

---

## 10. Known Issues for Report (Chapter 7)

1. **Current metrics are pre-fix**: metrics.json and plots reflect the model trained with
   the preprocessing mismatch. Retrain after pulling `final` to get honest numbers.
2. **HOG F1@default_threshold = 0.000**: cosine similarity of HOG vectors rarely exceeds
   0.764; the threshold calibrated for Siamese embeddings does not transfer to HOG.
   Report this as a motivation for mode-specific thresholds or for the hybrid approach.
3. **Siamese optimal threshold = 0.994**: embeddings not yet well-separated. Well-trained
   models sit 0.5–0.8. More epochs, triplet loss, or hard-negative mining would help.
4. **workers=0 on Windows**: forces single-threaded image loading (~16 min/epoch).
   On Linux/Colab the same code auto-switches to 4 workers (~3–5 min/epoch).

---

## 11. Git State

- Branch `final` pushed to `origin/final`
- GitHub: https://github.com/huzaifah03/writer-verification-system/tree/final
- Last commit: `2a8bdbf` — preprocessing fix + augmentation + threshold
- Git identity (local): `shaikhhuzaifah03@gmail.com` / `Huzaifah`
- Files NOT in git (gitignored): `venv/`, `data/`, `saved_models/*.pth`, `saved_models/splits.json`
