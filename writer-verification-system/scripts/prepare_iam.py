"""
scripts/prepare_iam.py — Convert IAM Handwriting Database into per-writer folders.

Reads:
    <iam_dir>/ascii/forms.txt   — form → writer mapping
    <iam_dir>/lines/**/*.png    — line images

Writes:
    <out_dir>/writer_<ID>/      — one folder per writer, images copied in

Usage:
    python scripts/prepare_iam.py --iam_dir data/iam --out_dir data/processed --min_samples 5
"""

import argparse
import os
import shutil


def parse_forms(forms_txt):
    """Return {form_id: writer_id} from ascii/forms.txt."""
    mapping = {}
    with open(forms_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                form_id, writer_id = parts[0], parts[1]
                mapping[form_id] = writer_id
    return mapping


def find_line_images(lines_dir, form_id):
    """
    Return list of PNG paths for a given form.
    IAM structure: lines/<prefix>/<form_id>/<line_id>.png
    where <prefix> is the part of form_id before the first '-'.
    Some archives place everything flat under lines/ — we search both.
    """
    prefix = form_id.split("-")[0]
    candidates = [
        os.path.join(lines_dir, prefix, form_id),
        os.path.join(lines_dir, form_id),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            return [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".png")
            ]
    return []


def main():
    parser = argparse.ArgumentParser(description="Prepare IAM dataset for training")
    parser.add_argument("--iam_dir", type=str, default="data/iam")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum line images per writer to include")
    args = parser.parse_args()

    forms_txt = os.path.join(args.iam_dir, "ascii", "forms.txt")
    lines_dir = os.path.join(args.iam_dir, "lines")

    if not os.path.isfile(forms_txt):
        raise FileNotFoundError(f"forms.txt not found at {forms_txt}")
    if not os.path.isdir(lines_dir):
        raise FileNotFoundError(f"lines/ directory not found at {lines_dir}")

    form_to_writer = parse_forms(forms_txt)
    print(f"Parsed {len(form_to_writer)} forms from forms.txt")

    # Accumulate images per writer
    writer_images = {}
    missing_forms = 0
    for form_id, writer_id in form_to_writer.items():
        imgs = find_line_images(lines_dir, form_id)
        if not imgs:
            missing_forms += 1
            continue
        writer_images.setdefault(writer_id, []).extend(imgs)

    if missing_forms:
        print(f"  Warning: {missing_forms} forms had no images in lines/ (normal if forms*.tgz not extracted)")

    # Filter by min_samples
    before = len(writer_images)
    writer_images = {w: imgs for w, imgs in writer_images.items() if len(imgs) >= args.min_samples}
    print(f"Writers with >= {args.min_samples} samples: {len(writer_images)} / {before}")

    # Copy images to out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    total_images = 0
    for writer_id, imgs in sorted(writer_images.items()):
        writer_dir = os.path.join(args.out_dir, f"writer_{writer_id.zfill(3)}")
        os.makedirs(writer_dir, exist_ok=True)
        for src in imgs:
            dst = os.path.join(writer_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        total_images += len(imgs)

    print(f"Done. Wrote {total_images} images across {len(writer_images)} writer folders -> {args.out_dir}")


if __name__ == "__main__":
    main()
