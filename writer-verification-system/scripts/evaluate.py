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
