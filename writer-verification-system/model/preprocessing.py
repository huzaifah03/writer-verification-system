"""
Image Preprocessing Module
---------------------------
preprocess_for_model: resize -> 3-ch grayscale -> ToTensor -> ImageNet normalise.
  Matches WriterPairDataset's training transform exactly to avoid distribution mismatch.

preprocess_for_visualization: retains the Gaussian+Otsu pipeline for display purposes only.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from config import Config


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as a NumPy array (BGR)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def binarize(img: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to binarize the image.
    Works best on grayscale images after denoising.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def normalize_and_resize(img: np.ndarray) -> np.ndarray:
    """
    Resize to the model's expected input size (224x224).
    Converts binary/grayscale back to 3-channel for ResNet50.
    """
    target_size = Config.IMAGE_SIZE  # (224, 224)
    resized = cv2.resize(img, target_size)

    # ResNet50 expects 3-channel RGB input
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return resized


def preprocess_for_model(image_path: str):
    """
    Preprocessing pipeline aligned with training transforms → returns a PyTorch
    tensor ready for the model.

    Pipeline: open → resize 224×224 → 3-channel grayscale → ToTensor → Normalize.
    Must match WriterPairDataset's transform exactly; divergence causes a
    train/inference distribution mismatch that suppresses AUC.

    Args:
        image_path: Path to the uploaded handwriting image.

    Returns:
        torch.Tensor of shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if image_path.lower().endswith(".pdf"):
        import fitz
        doc = fitz.open(image_path)
        pix = doc[0].get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)


def preprocess_for_visualization(image_path: str) -> np.ndarray:
    """
    Preprocessing pipeline that returns a numpy array for display purposes.
    (Used by the frontend to show the processed image.)
    """
    img = load_image(image_path)
    img = to_grayscale(img)
    img = denoise(img)
    img = binarize(img)
    img = normalize_and_resize(img)
    return img
