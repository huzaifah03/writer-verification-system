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
