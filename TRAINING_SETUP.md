# Training PC Setup & Run Guide — Writer Verification System (FYP-2)

**Audience:** Claude Code, running on the GPU / training machine.
**Goal of this session:** Set up the environment, add the evaluation pipeline,
prepare the IAM dataset, **train** the writer-verification model, and **evaluate**
it — producing real metrics and plots for the project report (Chapter 7).

Work through the steps in order. **Stop and report to the human** if any step
fails or any prerequisite is missing. Ask before destructive actions.

---

## 0. Project context (read first)

This is an AI-based handwriting writer-verification system for an undergraduate
FYP. Two handwritten samples are compared and the system decides whether they
were written by the same person.

**Pipeline (already implemented in the repo):**
- `model/feature_extractor.py` — a Siamese embedding network (`WriterEmbeddingNet`,
  ResNet-50 backbone + 256-dim projection head) AND a HOG extractor.
- Inference combines a 256-dim Siamese embedding with a 26,244-dim HOG vector,
  L2-normalizes each, concatenates (26,500-dim), and compares with cosine
  similarity (`model/predict.py`, `model/similarity.py`).
- The model has **not been trained yet**. That is the purpose of this session.

**What is missing from the cloned repo and must be created in Step 2:**
- `training/__init__.py`
- `training/data_split.py` (deterministic writer-level train/val/test split)
- an updated `training/train.py` (uses the split, saves it for reproducibility)
- `scripts/evaluate.py` (evaluation + plots)

The exact contents are embedded in Step 2 below.

---

## 1. Environment setup

### 1.1 Confirm you are on the right branch
```bash
git branch --show-current
git log --oneline -3
```
Expected: the branch containing the latest code (the one the human cloned —
likely `new-branch`). If on a different branch, ask the human which to use.
Do **not** switch branches without confirmation.

### 1.2 Create and activate a virtual environment
```bash
python --version            # confirm Python 3.10+ (3.12 used on dev machine)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 1.3 Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 CRITICAL — verify GPU / CUDA is available
The dev machine used a CPU-only PyTorch build. **This machine has a GPU**, so we
need the CUDA build. After install, verify:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
- If it prints `CUDA available: True` → good, continue.
- If it prints `False` → the CPU wheel got installed. Reinstall the CUDA build
  (do NOT use the cpu index URL). For example, for CUDA 12.1:
  ```bash
  pip uninstall -y torch torchvision
  pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
  ```
  Then re-run the verification. Ask the human which CUDA version their driver
  supports if unsure (`nvidia-smi` shows the driver's max CUDA version).

---

## 2. Add the evaluation pipeline files

Create the following four files **exactly** as specified. These are not yet in
the repo.

### 2.1 `training/__init__.py`
Create an **empty** file at `training/__init__.py` (it makes `training` an
importable package).

### 2.2 `training/data_split.py`
```python
"""
Deterministic writer-level train / val / test split.

Shared by training/train.py and scripts/evaluate.py so that the test set is
GUARANTEED disjoint from the training and validation writers, and the split is
fully reproducible across runs.

Splitting at the writer level (not the pair level) is essential for valid
writer-verification evaluation: if any image of a writer appears in training,
the model has effectively "seen" that writer, and test scores would be
optimistically biased. Here, an entire writer is assigned to exactly one split.
"""

import os
import random


def get_writer_splits(data_dir: str,
                      seed: int = 42,
                      val_frac: float = 0.15,
                      test_frac: float = 0.15):
    """
    Return (train_dirs, val_dirs, test_dirs) as lists of absolute writer-folder
    paths under `data_dir`.

    The writer folders are sorted before shuffling so the split depends only on
    the folder names and the seed — not on filesystem iteration order — making
    it reproducible on any machine.

    Args:
        data_dir:  Path containing one sub-directory per writer
                   (e.g. data/processed/writer_001, writer_002, ...).
        seed:      RNG seed for the shuffle. Keep this identical in train.py
                   and evaluate.py.
        val_frac:  Fraction of writers held out for validation.
        test_frac: Fraction of writers held out for the final test set.

    Returns:
        (train_dirs, val_dirs, test_dirs)
    """
    writer_dirs = sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    rng = random.Random(seed)
    rng.shuffle(writer_dirs)

    n = len(writer_dirs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_dirs = writer_dirs[:n_test]
    val_dirs = writer_dirs[n_test:n_test + n_val]
    train_dirs = writer_dirs[n_test + n_val:]

    return train_dirs, val_dirs, test_dirs
```

### 2.3 `training/train.py` (REPLACE the existing file)
The existing `training/train.py` does a 2-way split. Replace its entire contents
with the version below, which does a reproducible 3-way writer-level split and
records it to `saved_models/splits.json`:
```python
"""
Training Script — Writer Verification Model
---------------------------------------------
Trains a ResNet50-based Siamese-style network on handwriting image pairs.
Supports local GPU (e.g. GTX 1660 Ti) and can be adapted for Colab/Kaggle.

Usage:
    python training/train.py --data_dir data/processed --epochs 20 --batch_size 32

Dataset expected structure:
    data/processed/
        writer_001/
            sample_01.png
            sample_02.png
        writer_002/
            ...

The script generates pairs (same writer / different writer) automatically and
uses a writer-level train/val/test split (see training/data_split.py). The test
writers are held out entirely from training and are recorded in
saved_models/splits.json so that scripts/evaluate.py can evaluate on exactly the
same unseen writers.
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model.feature_extractor import WriterEmbeddingNet
from training.data_split import get_writer_splits


# ---------------------------------------------------------------------------
# Dataset — Pair Generator
# ---------------------------------------------------------------------------

class WriterPairDataset(Dataset):
    """
    Generates (image1, image2, label) triplets.
    label = 1 -> Same Writer, label = 0 -> Different Writer
    """

    def __init__(self, writer_dirs: list, transform=None, pairs_per_writer: int = 10):
        self.transform = transform
        self.pairs = []

        all_writers = {d: self._get_images(d) for d in writer_dirs}
        writer_list = list(all_writers.keys())

        for writer, images in all_writers.items():
            if len(images) < 2:
                continue
            # Positive pairs (same writer)
            for _ in range(pairs_per_writer):
                i1, i2 = random.sample(images, 2)
                self.pairs.append((i1, i2, 1))

            # Negative pairs (different writer)
            for _ in range(pairs_per_writer):
                other_writer = random.choice([w for w in writer_list if w != writer])
                other_images = all_writers[other_writer]
                if other_images:
                    i1 = random.choice(images)
                    i2 = random.choice(other_images)
                    self.pairs.append((i1, i2, 0))

        random.shuffle(self.pairs)

    def _get_images(self, writer_dir: str) -> list:
        exts = {".png", ".jpg", ".jpeg"}
        return [
            os.path.join(writer_dir, f)
            for f in os.listdir(writer_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model — WriterEmbeddingNet imported from model.feature_extractor
# (single source of truth — architecture lives there, used by both
#  training and inference)
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese training.
    Pulls same-writer pairs together, pushes different-writer pairs apart.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss = label * dist.pow(2) + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data transforms (training-time augmentation kept minimal for reproducibility)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Writer-level 3-way split (train / val / test). Test writers are held out
    # entirely and only used later by scripts/evaluate.py.
    train_dirs, val_dirs, test_dirs = get_writer_splits(args.data_dir, seed=42)
    print(f"Writers — train: {len(train_dirs)}, val: {len(val_dirs)}, "
          f"test (held out, not used in training): {len(test_dirs)}")

    # Persist the split so evaluation uses exactly the same held-out writers.
    os.makedirs("saved_models", exist_ok=True)
    with open(os.path.join("saved_models", "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": 42, "train": train_dirs, "val": val_dirs, "test": test_dirs},
                  f, indent=2)
    print("Saved writer split to saved_models/splits.json")

    train_dataset = WriterPairDataset(train_dirs, transform=transform)
    val_dataset = WriterPairDataset(val_dirs, transform=transform, pairs_per_writer=5)

    # num_workers > 0 crashes on Windows unless inside if __name__ == "__main__"
    workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    # Model, loss, optimizer
    model = WriterEmbeddingNet().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            emb1, emb2 = model(img1), model(img2)
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                emb1, emb2 = model(img1), model(img2)
                val_loss += criterion(emb1, emb2, labels).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "saved_models/writer_verification_model.pth")
            print(f"  Best model saved (val_loss={val_loss:.4f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train writer verification model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
```

### 2.4 `scripts/evaluate.py` (NEW file)
```python
"""
scripts/evaluate.py — Evaluate the trained writer verification model.

Evaluates on a writer-level held-out TEST set (the writers recorded in
saved_models/splits.json by training/train.py, or recomputed deterministically
from the same seed if that file is absent). No test writer is ever seen during
training, so the reported metrics are an honest estimate of generalisation.

The script evaluates the ACTUAL deployed similarity pipeline (hybrid Siamese +
HOG cosine similarity, exactly as model/predict.py uses it) and additionally
runs two ablations — Siamese-only and HOG-only — so the contribution of each
feature stream can be discussed in the report.

Outputs (written to results/):
    metrics.json            — all numeric metrics for every mode and threshold
    roc_curve.png           — ROC curves for hybrid / Siamese-only / HOG-only
    score_distribution.png  — score histograms (same-writer vs different-writer)
    confusion_matrix.png    — hybrid confusion matrices at default + optimal thr

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --data_dir data/processed --pairs_per_writer 10
"""

import argparse
import json
import os
import random
import sys

import numpy as np

# Ensure project root is importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # headless — write PNGs without a display
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)

from config import Config
from model.preprocessing import preprocess_for_model
from model.feature_extractor import (
    get_siamese_extractor, extract_siamese_features, extract_hog_features,
)
from model.similarity import cosine_similarity
from training.data_split import get_writer_splits

IMG_EXTS = {".png", ".jpg", ".jpeg"}
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# Test-set construction
# ---------------------------------------------------------------------------

def list_images(writer_dir):
    return [
        os.path.join(writer_dir, f)
        for f in os.listdir(writer_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]


def load_test_writers(data_dir):
    """Prefer the split saved at training time; otherwise recompute it."""
    splits_path = os.path.join("saved_models", "splits.json")
    if os.path.isfile(splits_path):
        with open(splits_path, encoding="utf-8") as f:
            data = json.load(f)
        test_dirs = data.get("test", [])
        # Keep only paths that still exist on this machine
        test_dirs = [d for d in test_dirs if os.path.isdir(d)]
        if test_dirs:
            print(f"Loaded {len(test_dirs)} test writers from {splits_path}")
            return test_dirs
        print("splits.json had no usable test paths; recomputing split.")
    _, _, test_dirs = get_writer_splits(data_dir, seed=42)
    print(f"Recomputed split — {len(test_dirs)} test writers (seed=42)")
    return test_dirs


def build_test_pairs(test_dirs, pairs_per_writer, seed=42):
    """Balanced set of same-writer (label 1) and different-writer (label 0) pairs."""
    rng = random.Random(seed)
    writer_images = {d: list_images(d) for d in test_dirs}
    writer_images = {d: imgs for d, imgs in writer_images.items() if len(imgs) >= 2}
    writers = list(writer_images.keys())

    if len(writers) < 2:
        sys.exit("ERROR: need at least 2 test writers with 2+ images each to evaluate.")

    pairs = []
    for w in writers:
        imgs = writer_images[w]
        for _ in range(pairs_per_writer):                      # positive pairs
            a, b = rng.sample(imgs, 2)
            pairs.append((a, b, 1))
        for _ in range(pairs_per_writer):                      # negative pairs
            other = rng.choice([x for x in writers if x != w])
            a = rng.choice(imgs)
            b = rng.choice(writer_images[other])
            pairs.append((a, b, 0))
    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Feature caching + scoring
# ---------------------------------------------------------------------------

def cache_features(pairs, device):
    """
    Compute the L2-normalised Siamese (256-d) and HOG (26244-d) feature vectors
    once per unique image, mirroring extract_siamese_hybrid_features exactly.
    """
    unique = sorted({p for pair in pairs for p in pair[:2]})
    print(f"Extracting features for {len(unique)} unique images "
          f"(device={device})...")
    extractor = get_siamese_extractor(device, Config.MODEL_PATH)

    siamese_cache, hog_cache = {}, {}
    for i, path in enumerate(unique, 1):
        tensor = preprocess_for_model(path)
        s = extract_siamese_features(tensor, extractor, device)
        h = extract_hog_features(path)
        s = s / (np.linalg.norm(s) + 1e-8)
        h = h / (np.linalg.norm(h) + 1e-8)
        siamese_cache[path] = s
        hog_cache[path] = h
        if i % 25 == 0 or i == len(unique):
            print(f"  {i}/{len(unique)} images done")
    return siamese_cache, hog_cache


def score_all(pairs, siamese_cache, hog_cache):
    """Return {mode: np.array(scores)} for hybrid / siamese / hog, plus labels."""
    modes = {"hybrid": [], "siamese": [], "hog": []}
    labels = []
    for a, b, label in pairs:
        sa, sb = siamese_cache[a], siamese_cache[b]
        ha, hb = hog_cache[a], hog_cache[b]
        modes["siamese"].append(cosine_similarity(sa, sb))
        modes["hog"].append(cosine_similarity(ha, hb))
        modes["hybrid"].append(
            cosine_similarity(np.concatenate([sa, ha]), np.concatenate([sb, hb])))
        labels.append(label)
    return {k: np.asarray(v) for k, v in modes.items()}, np.asarray(labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_at(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }


def youden_optimal(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx]), float(roc_auc), fpr, tpr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(roc_data, out_path):
    plt.figure(figsize=(6, 6))
    for mode, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{mode} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Writer Verification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_score_distribution(scores, labels, out_path):
    plt.figure(figsize=(7, 4.5))
    same = scores[labels == 1]
    diff = scores[labels == 0]
    bins = np.linspace(0, 1, 30)
    plt.hist(diff, bins=bins, alpha=0.6, label="Different writer")
    plt.hist(same, bins=bins, alpha=0.6, label="Same writer")
    plt.axvline(Config.SIMILARITY_THRESHOLD, color="red", linestyle="--",
                label=f"Threshold = {Config.SIMILARITY_THRESHOLD}")
    plt.xlabel("Cosine similarity score")
    plt.ylabel("Number of pairs")
    plt.title("Score Distribution (Hybrid) — Same vs Different Writer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm_default, cm_optimal, thr_default, thr_optimal, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cm, thr, title in [
        (axes[0], cm_default, thr_default, "Default threshold"),
        (axes[1], cm_optimal, thr_optimal, "Optimal threshold"),
    ]:
        cm = np.asarray(cm)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{title} ({thr:.3f})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Diff", "Same"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Diff", "Same"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for r in range(2):
            for col in range(2):
                ax.text(col, r, str(cm[r, col]), ha="center", va="center",
                        color="black", fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrix (Hybrid)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate writer verification model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--pairs_per_writer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(Config.MODEL_PATH):
        print(f"WARNING: no trained model at {Config.MODEL_PATH}. "
              "Results will reflect an UNTRAINED model and are not meaningful.")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    test_dirs = load_test_writers(args.data_dir)
    pairs = build_test_pairs(test_dirs, args.pairs_per_writer, seed=args.seed)
    n_pos = sum(1 for _, _, y in pairs if y == 1)
    print(f"Built {len(pairs)} test pairs ({n_pos} same-writer, "
          f"{len(pairs) - n_pos} different-writer)")

    siamese_cache, hog_cache = cache_features(pairs, device)
    scores_by_mode, labels = score_all(pairs, siamese_cache, hog_cache)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    report = {
        "n_test_writers": len(test_dirs),
        "n_pairs": len(pairs),
        "n_same": int(n_pos),
        "n_different": int(len(pairs) - n_pos),
        "default_threshold": Config.SIMILARITY_THRESHOLD,
        "modes": {},
    }
    roc_data = {}

    for mode, scores in scores_by_mode.items():
        thr_opt, roc_auc, fpr, tpr = youden_optimal(scores, labels)
        roc_data[mode] = (fpr, tpr, roc_auc)
        report["modes"][mode] = {
            "auc": round(roc_auc, 4),
            "at_default_threshold": metrics_at(scores, labels, Config.SIMILARITY_THRESHOLD),
            "at_optimal_threshold": metrics_at(scores, labels, thr_opt),
        }

    # Plots (hybrid is the headline; ROC overlays all three)
    plot_roc(roc_data, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_score_distribution(scores_by_mode["hybrid"], labels,
                            os.path.join(RESULTS_DIR, "score_distribution.png"))
    hybrid = report["modes"]["hybrid"]
    plot_confusion(
        hybrid["at_default_threshold"]["confusion_matrix"],
        hybrid["at_optimal_threshold"]["confusion_matrix"],
        hybrid["at_default_threshold"]["threshold"],
        hybrid["at_optimal_threshold"]["threshold"],
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Test writers: {report['n_test_writers']}   Pairs: {report['n_pairs']} "
          f"({report['n_same']} same / {report['n_different']} different)")
    print(f"{'Mode':<10}{'AUC':>8}{'Acc@0.75':>11}{'F1@0.75':>10}"
          f"{'Acc@opt':>10}{'F1@opt':>9}{'opt_thr':>9}")
    for mode in ["hybrid", "siamese", "hog"]:
        m = report["modes"][mode]
        d = m["at_default_threshold"]
        o = m["at_optimal_threshold"]
        print(f"{mode:<10}{m['auc']:>8.3f}{d['accuracy']:>11.3f}{d['f1']:>10.3f}"
              f"{o['accuracy']:>10.3f}{o['f1']:>9.3f}{o['threshold']:>9.3f}")
    print("====================================================")
    print(f"\nWrote: {RESULTS_DIR}/metrics.json + 3 PNG plots")


if __name__ == "__main__":
    main()
```

### 2.5 Verify the imports resolve
```bash
python -c "from training.data_split import get_writer_splits; print('split import OK')"
python -c "import ast; ast.parse(open('scripts/evaluate.py').read()); print('evaluate.py parses OK')"
```
Both must succeed before continuing.

---

## 3. Prepare the IAM dataset

The training data is the **IAM Handwriting Database**. The `scripts/prepare_iam.py`
script (already in the repo) converts it into the per-writer folder structure the
trainer expects (`data/processed/writer_XXX/*.png`).

### 3.1 Locate the IAM archives
**Ask the human where the IAM `.tgz` files are on this machine.** They previously
downloaded these five files:
`ascii.tgz`, `lines.tgz`, `formsA-D.tgz`, `formsE-H.tgz`, `formsI-Z.tgz`.

For training, **only `ascii.tgz` and `lines.tgz` are required** (the prepare
script uses `ascii/forms.txt` for the form→writer mapping and `lines/` for the
images). The `forms*.tgz` files are not needed for this pipeline.

If the archives are not on this machine, ask the human to copy `ascii.tgz` and
`lines.tgz` over from the dev machine (USB, network share, or cloud).

### 3.2 Extract into `data/iam/`
Create the target structure `data/iam/ascii/` and `data/iam/lines/`:
```bash
# From the project root. Adjust the source path to wherever the archives live.
mkdir -p data/iam
tar -xzf <path>/ascii.tgz -C data/iam/ascii    # may need: mkdir -p data/iam/ascii first
tar -xzf <path>/lines.tgz -C data/iam/lines    # may need: mkdir -p data/iam/lines first
```
Note: some IAM archives extract a nested folder; the goal is for these paths to
exist afterwards:
- `data/iam/ascii/forms.txt`
- `data/iam/lines/<prefix>/<form_id>/<line_id>.png`

Verify:
```bash
test -f data/iam/ascii/forms.txt && echo "forms.txt found" || echo "MISSING forms.txt"
python -c "import glob; print('line images:', len(glob.glob('data/iam/lines/**/*.png', recursive=True)))"
```
The line-image count should be in the thousands. If it is 0, the extraction path
is wrong — inspect the actual folder layout and adjust.

### 3.3 Run the prepare script
```bash
python scripts/prepare_iam.py --iam_dir data/iam --out_dir data/processed --min_samples 5
```
This writes `data/processed/writer_XXX/` folders. Confirm:
```bash
python -c "import os; d='data/processed'; print('writers:', len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))]))"
```
Expect on the order of a few hundred writers. Report the count to the human.

---

## 4. Smoke test (1 epoch) — validate the whole pipeline first

Before the full run, confirm the pipeline works end-to-end with a tiny run:
```bash
python training/train.py --data_dir data/processed --epochs 1 --batch_size 16
```
Success criteria:
- prints the train/val/test writer counts,
- writes `saved_models/splits.json`,
- completes one epoch without error,
- writes `saved_models/writer_verification_model.pth`.

If the smoke test passes, continue to the full run. If it errors (out-of-memory,
data loading, etc.), report the error before proceeding. For GPU out-of-memory,
reduce `--batch_size` (try 16, then 8).

---

## 5. Full training run

```bash
python training/train.py --data_dir data/processed --epochs 20 --batch_size 32
```
- Watch that **Val Loss** generally trends downward. The best model (lowest val
  loss) is saved automatically to `saved_models/writer_verification_model.pth`.
- Note the final train/val losses to report back.
- On a mid-range GPU this may take roughly 1–3 hours depending on dataset size.
  If time-constrained, 10 epochs is acceptable for a first result.

---

## 6. Evaluation — produce the report metrics

```bash
python scripts/evaluate.py --data_dir data/processed --pairs_per_writer 10
```
This evaluates on the **held-out test writers** (never seen in training) and
writes to `results/`:
- `metrics.json` — AUC, accuracy, precision, recall, F1 for the hybrid pipeline
  plus Siamese-only and HOG-only ablations, at the default 0.75 threshold and at
  the ROC-optimal threshold.
- `roc_curve.png`, `score_distribution.png`, `confusion_matrix.png`.

It also prints a summary table to the console. Capture that table.

---

## 7. Report back to the human

Provide:
1. The console summary table from Step 6.
2. The contents of `results/metrics.json`.
3. The three PNG files in `results/`.
4. The final train/val loss from Step 5 and the writer count from Step 3.3.
5. Copy these back to the dev machine (needed for the live app + the report):
   - `saved_models/writer_verification_model.pth`
   - `saved_models/splits.json`
   - the entire `results/` folder.

The human will use the metrics and plots to write Chapter 7 (Results & Discussion).

---

## 8. Known issues to watch and report on

- **Preprocessing mismatch (important).** Training (`training/train.py`) feeds the
  model grayscale+resize images, but inference (`model/preprocessing.py`,
  `preprocess_for_model`) additionally applies Gaussian denoise + Otsu
  binarization. The model is therefore trained and evaluated on slightly
  different image distributions. If the **hybrid** and **Siamese-only** accuracy
  in the evaluation come back poor (near chance, ~0.5 AUC), this mismatch is the
  prime suspect. Report the numbers; do **not** silently change the preprocessing
  — flag it and let the human decide whether to align them and retrain.
- **HOG-only as a sanity check.** HOG uses no trained weights, so its metrics
  should be stable regardless of training. If HOG-only outperforms the hybrid by
  a wide margin, that is a strong signal the Siamese half is hurting rather than
  helping (again, likely the preprocessing mismatch or insufficient training).
- **GPU memory.** If you hit CUDA out-of-memory, lower `--batch_size`.
- **Reproducibility.** Both the trainer and evaluator use seed 42 and the same
  `get_writer_splits`, so the test writers are identical across runs. Do not
  change the seed between training and evaluation.

---

## 9. Do NOT

- Do not commit `data/iam/`, `data/processed/`, `venv/`, or `*.pth` to git (they
  are large and/or gitignored). The small `results/` folder MAY be committed if
  the human wants the metrics preserved in the repo.
- Do not change model architecture, thresholds, or preprocessing without asking.
- Do not switch git branches or force-push.
