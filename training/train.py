"""
Training Script — Writer Verification Model
---------------------------------------------
Trains a ResNet50-based Siamese-style network on handwriting image pairs.
Supports local GPU (e.g. GTX 1660 Ti) and can be adapted for Colab/Kaggle.

Usage:
    python training/train.py --data_dir data/processed --epochs 20 --batch_size 32

Dataset expected structure:
    data/
        writer_001/
            sample_01.png
            sample_02.png
        writer_002/
            ...

The script generates pairs (same writer / different writer) automatically.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from model.feature_extractor import WriterEmbeddingNet


# ---------------------------------------------------------------------------
# Dataset — Pair Generator
# ---------------------------------------------------------------------------

class WriterPairDataset(Dataset):
    """
    Generates (image1, image2, label) triplets.
    label = 1 → Same Writer, label = 0 → Different Writer
    """

    def __init__(self, writer_dirs: list, transform=None, pairs_per_writer: int = 10):
        self.transform = transform
        self.pairs = []

        # Build pairs
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

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Discover writer directories
    writer_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]
    print(f"Found {len(writer_dirs)} writers.")

    train_dirs, val_dirs = train_test_split(writer_dirs, test_size=0.2, random_state=42)

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
    os.makedirs("saved_models", exist_ok=True)

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

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "saved_models/writer_verification_model.pth")
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

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
