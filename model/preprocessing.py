"""
Image Preprocessing Module
---------------------------
Applies the following pipeline to each uploaded handwriting image:
1. Load image
2. Convert to grayscale
3. Denoise (Gaussian blur)
4. Binarize (Otsu thresholding)
5. Normalize and resize for ResNet50 input
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from config import Config


def _pdf_to_bgr(path: str) -> np.ndarray:
    """
    Render the first page of a PDF to a BGR numpy array.
    Uses PyMuPDF (fitz) — no external binaries required.
    Renders at 2× scale so fine handwriting strokes survive the
    subsequent resize to 224×224.
    """
    import fitz  # pymupdf — imported lazily so non-PDF paths pay no cost
    doc = fitz.open(path)
    page = doc[0]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as a NumPy array (BGR).
    Accepts JPEG, PNG, and single-page PDF inputs.
    """
    if image_path.lower().endswith(".pdf"):
        return _pdf_to_bgr(image_path)
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
    Full preprocessing pipeline → returns a PyTorch tensor ready for the model.

    Args:
        image_path: Path to the uploaded handwriting image.

    Returns:
        torch.Tensor of shape (1, 3, 224, 224)
    """
    img = load_image(image_path)
    img = to_grayscale(img)
    img = denoise(img)
    img = binarize(img)
    img = normalize_and_resize(img)

    # Convert to PIL for torchvision transforms
    pil_img = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # ImageNet normalization (standard for ResNet50 pretrained weights)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension → (1, 3, 224, 224)
    return tensor


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
