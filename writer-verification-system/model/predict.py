"""
Inference Entry Point
----------------------
This module is the single entry point called by the Flask backend.
It orchestrates the full pipeline:
    image path → preprocess → extract features → compare → result dict
"""

import torch
from config import Config
from model.preprocessing import preprocess_for_model
from model.feature_extractor import get_siamese_extractor, extract_siamese_features
from model.similarity import compare_features


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def verify_writers(image_path1: str, image_path2: str) -> dict:
    """
    Run the full writer verification pipeline on two handwriting images.

    Args:
        image_path1: Absolute path to the first uploaded assignment image.
        image_path2: Absolute path to the second uploaded assignment image.

    Returns:
        dict with:
            - similarity_score     (float, 0.0–1.0)
            - similarity_percentage (float, 0.0–100.0)
            - decision             (str: "Same Writer" / "Different Writer")
            - risk_level           (str: "Low" / "Medium" / "High")

    Raises:
        FileNotFoundError: if either image path does not exist.
        RuntimeError: if feature extraction fails.
    """
    device = get_device()
    tensor1 = preprocess_for_model(image_path1)
    tensor2 = preprocess_for_model(image_path2)
    extractor = get_siamese_extractor(device, Config.MODEL_PATH)
    f1 = extract_siamese_features(tensor1, extractor, device)
    f2 = extract_siamese_features(tensor2, extractor, device)
    return compare_features(f1, f2)
