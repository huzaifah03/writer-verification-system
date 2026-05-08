"""
Similarity Measurement Module
-------------------------------
Computes a similarity score between two handwriting feature vectors
and produces a final verification decision.

Method: Cosine Similarity
- Returns a value between 0 (completely different) and 1 (identical).
- Threshold (Config.SIMILARITY_THRESHOLD) determines Same/Different Writer.
- Risk level is mapped from the score for reporting purposes.
"""

import numpy as np
from scipy.spatial.distance import cosine
from config import Config


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    Returns:
        Float in range [0, 1]. Higher = more similar.
    """
    sim = 1 - cosine(vec1, vec2)
    # Clamp to [0, 1] to handle floating point edge cases
    return float(np.clip(sim, 0.0, 1.0))


def get_decision(similarity_score: float) -> str:
    """
    Convert a similarity score to a Same/Different Writer decision.

    Args:
        similarity_score: Float in [0, 1]

    Returns:
        "Same Writer" or "Different Writer"
    """
    threshold = Config.SIMILARITY_THRESHOLD
    return "Same Writer" if similarity_score >= threshold else "Different Writer"


def get_risk_level(similarity_score: float) -> str:
    """
    Map similarity score to a plagiarism risk level for the report.

    Thresholds (to be calibrated after model evaluation):
        >= 0.85  → High
        >= 0.65  → Medium
        <  0.65  → Low

    Args:
        similarity_score: Float in [0, 1]

    Returns:
        "High", "Medium", or "Low"
    """
    if similarity_score >= 0.85:
        return "High"
    elif similarity_score >= 0.65:
        return "Medium"
    else:
        return "Low"


def compare_features(features1: np.ndarray, features2: np.ndarray) -> dict:
    """
    Full comparison pipeline: compute score → decision → risk level.

    Args:
        features1: Hybrid feature vector for assignment 1
        features2: Hybrid feature vector for assignment 2

    Returns:
        dict with keys:
            - similarity_score (float, 0–1)
            - similarity_percentage (float, 0–100)
            - decision (str)
            - risk_level (str)
    """
    score = cosine_similarity(features1, features2)
    decision = get_decision(score)
    risk = get_risk_level(score)

    return {
        "similarity_score": round(score, 4),
        "similarity_percentage": round(score * 100, 2),
        "decision": decision,
        "risk_level": risk,
    }
