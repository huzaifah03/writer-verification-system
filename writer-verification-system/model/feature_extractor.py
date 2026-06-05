"""
Hybrid Feature Extraction Module
----------------------------------
Combines:
1. Deep learning features — ResNet50 (pretrained on ImageNet), last FC layer removed,
   outputs a 2048-dim embedding vector per image.
2. Traditional handwriting features — HOG (Histogram of Oriented Gradients),
   capturing stroke direction and texture patterns.

Final feature vector = concatenation of CNN features + HOG features.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from skimage.feature import hog
from skimage import color
import cv2
from config import Config


# ---------------------------------------------------------------------------
# CNN Feature Extractor (ResNet50)
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    """
    ResNet50 with the final classification layer removed.
    Outputs a 2048-dimensional feature vector (global average pooled).
    """

    def __init__(self, pretrained: bool = True):
        super(CNNFeatureExtractor, self).__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Remove the final FC layer → use as feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 3, 224, 224)
        Returns:
            Feature vector of shape (batch, 2048)
        """
        with torch.no_grad():
            features = self.features(x)
        return features.view(features.size(0), -1)  # Flatten → (batch, 2048)


def extract_cnn_features(image_tensor: torch.Tensor,
                         extractor: CNNFeatureExtractor,
                         device: str = "cpu") -> np.ndarray:
    """
    Run the CNN extractor on a preprocessed image tensor.

    Args:
        image_tensor: Shape (1, 3, 224, 224) from preprocessing.py
        extractor: Instantiated CNNFeatureExtractor
        device: "cuda" or "cpu"

    Returns:
        1D numpy array of shape (2048,)
    """
    extractor.eval()
    extractor.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        features = extractor(image_tensor)

    return features.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Traditional Feature Extractor (HOG)
# ---------------------------------------------------------------------------

def extract_hog_features(image_path: str) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image.
    HOG captures stroke direction and local texture — useful for handwriting style.

    Args:
        image_path: Path to the original (or preprocessed) image.

    Returns:
        1D numpy array of HOG feature descriptors.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, Config.IMAGE_SIZE)
    img_gray = color.rgb2gray(img[:, :, ::-1])  # Convert BGR→RGB→gray for skimage

    features, _ = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    return features


# ---------------------------------------------------------------------------
# Combined Hybrid Feature Extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Siamese Embedding Network (trained for writer verification)
# ---------------------------------------------------------------------------

class WriterEmbeddingNet(nn.Module):
    """
    ResNet50 backbone with a 256-dim projection head.
    This is the trainable model — weights are learned via contrastive loss.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embed = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embed(x)


_siamese_extractor = None


def get_siamese_extractor(device: str = "cpu", model_path: str = None) -> WriterEmbeddingNet:
    """Return a cached WriterEmbeddingNet, loading weights if a path is given."""
    global _siamese_extractor
    if _siamese_extractor is None:
        _siamese_extractor = WriterEmbeddingNet()
        if model_path and os.path.isfile(model_path):
            state = torch.load(model_path, map_location=device)
            _siamese_extractor.load_state_dict(state)
        _siamese_extractor.eval()
        _siamese_extractor.to(device)
    return _siamese_extractor


def extract_siamese_features(image_tensor: torch.Tensor,
                              extractor: WriterEmbeddingNet,
                              device: str = "cpu") -> np.ndarray:
    """Run the Siamese net on a preprocessed image tensor, return 256-d numpy array."""
    extractor.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        embedding = extractor(image_tensor)
    return embedding.squeeze().cpu().numpy()


_cnn_extractor = None  # Singleton — loaded once per process


def get_cnn_extractor(device: str = "cpu") -> CNNFeatureExtractor:
    """Return a cached CNNFeatureExtractor instance."""
    global _cnn_extractor
    if _cnn_extractor is None:
        _cnn_extractor = CNNFeatureExtractor(pretrained=True)
        _cnn_extractor.eval()
    return _cnn_extractor


def extract_hybrid_features(image_path: str,
                             image_tensor: torch.Tensor,
                             device: str = "cpu") -> np.ndarray:
    """
    Extract and concatenate CNN + HOG features.

    Args:
        image_path: Original image path (for HOG)
        image_tensor: Preprocessed tensor from preprocessing.py (for CNN)
        device: "cuda" or "cpu"

    Returns:
        Combined 1D feature vector (2048 + HOG_dim,)
    """
    extractor = get_cnn_extractor(device)

    cnn_feats = extract_cnn_features(image_tensor, extractor, device)
    hog_feats = extract_hog_features(image_path)

    # Normalize each feature set to unit vector before concatenating
    cnn_feats = cnn_feats / (np.linalg.norm(cnn_feats) + 1e-8)
    hog_feats = hog_feats / (np.linalg.norm(hog_feats) + 1e-8)

    combined = np.concatenate([cnn_feats, hog_feats])
    return combined
