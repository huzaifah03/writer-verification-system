"""
Prepare IAM Handwriting Dataset for training.

Reads:  data/iam/ascii/forms.txt   (form_id → writer_id mapping)
        data/iam/lines/**/*.png    (segmented line images)

Writes: data/processed/writer_001/  (one folder per writer)
                        writer_002/
                        ...

Usage:
    python scripts/prepare_iam.py
    python scripts/prepare_iam.py --iam_dir data/iam --out_dir data/processed --min_samples 5

The training script expects:
    data/processed/<writer_dir>/<any_image>.png
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict

# Ensure project root is on sys.path when run from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Parser for ascii/forms.txt
# ---------------------------------------------------------------------------

def parse_forms_txt(forms_txt: str) -> dict:
    """
    Returns {form_id: writer_id} from forms.txt.

    forms.txt format (IAM):
        # comment lines start with #
        a01-000 000 36 57 34 ...
        ^form_id ^writer_id
    """
    mapping = {}
    with open(forms_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(iam_dir: str, out_dir: str, min_samples: int) -> None:
    forms_txt = os.path.join(iam_dir, "ascii", "forms.txt")
    lines_dir = os.path.join(iam_dir, "lines")

    # Validate inputs
    if not os.path.isfile(forms_txt):
        sys.exit(
            f"ERROR: forms.txt not found at {forms_txt}\n"
            "Make sure you extracted ascii.tgz into data/iam/"
        )
    if not os.path.isdir(lines_dir):
        sys.exit(
            f"ERROR: lines directory not found at {lines_dir}\n"
            "Make sure you extracted lines.tgz into data/iam/"
        )

    form_to_writer = parse_forms_txt(forms_txt)
    print(f"Loaded {len(form_to_writer):,} form → writer mappings from forms.txt")

    # Walk the lines/ directory tree.
    # Expected structure: lines/{prefix}/{form_id}/{line_id}.png
    # The form_id directory name is the key into form_to_writer.
    writer_images = defaultdict(list)  # writer_id → [src_path, ...]
    unmapped_forms = set()

    for entry in os.scandir(lines_dir):
        if not entry.is_dir():
            continue
        # entry = lines/a01/  (prefix dir)
        for form_entry in os.scandir(entry.path):
            if not form_entry.is_dir():
                continue
            form_id = form_entry.name          # e.g. "a01-000"
            writer_id = form_to_writer.get(form_id)
            if writer_id is None:
                unmapped_forms.add(form_id)
                continue
            for img_entry in os.scandir(form_entry.path):
                if img_entry.name.lower().endswith(".png"):
                    writer_images[writer_id].append(img_entry.path)

    if unmapped_forms:
        print(f"  Warning: {len(unmapped_forms)} forms had no writer mapping (skipped).")

    # Filter writers below minimum sample threshold
    before = len(writer_images)
    writer_images = {w: imgs for w, imgs in writer_images.items() if len(imgs) >= min_samples}
    dropped = before - len(writer_images)
    if dropped:
        print(f"  Dropped {dropped} writers with fewer than {min_samples} samples.")

    if not writer_images:
        sys.exit("ERROR: No writers found after filtering. Check your dataset path and extraction.")

    # Copy images into per-writer output folders
    print(f"\nCopying images for {len(writer_images):,} writers → {out_dir}")
    total_copied = 0

    for writer_id, src_paths in sorted(writer_images.items()):
        dest_dir = os.path.join(out_dir, f"writer_{writer_id.zfill(3)}")
        os.makedirs(dest_dir, exist_ok=True)
        for src in src_paths:
            shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
        total_copied += len(src_paths)

    # Summary
    print("\n" + "=" * 48)
    print("  IAM Dataset Preparation Complete")
    print("=" * 48)
    print(f"  Writers  : {len(writer_images):,}")
    print(f"  Images   : {total_copied:,}")
    print(f"  Output   : {os.path.abspath(out_dir)}")
    print("=" * 48)
    print("\nNext step:")
    print(f"  python training/train.py --data_dir {out_dir} --epochs 30 --batch_size 32")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IAM dataset for writer verification training")
    parser.add_argument(
        "--iam_dir", default="data/iam",
        help="Root of extracted IAM dataset (default: data/iam)"
    )
    parser.add_argument(
        "--out_dir", default="data/processed",
        help="Output directory for per-writer folders (default: data/processed)"
    )
    parser.add_argument(
        "--min_samples", type=int, default=5,
        help="Minimum line images a writer must have to be included (default: 5)"
    )
    args = parser.parse_args()
    build_dataset(args.iam_dir, args.out_dir, args.min_samples)
