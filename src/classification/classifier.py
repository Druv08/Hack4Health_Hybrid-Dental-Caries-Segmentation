"""
Binary Classifier for Dental Caries Detection
==============================================
Classifies dental X-rays as:
- CARIES (1): Dental caries/cavities detected
- NO CARIES (0): Healthy teeth, no cavities

Classification is based on features extracted from segmentation masks.

Author: Hack4Health Team
"""

import numpy as np
from typing import Dict, Tuple


class CariesClassifier:
    """
    Rule-based classifier for dental caries detection.
    
    Uses features from segmentation masks to make binary classification.
    Thresholds are calibrated for conservative medical screening.
    """
    
    def __init__(
        self,
        lesion_ratio_threshold: float = 0.001,
        min_component_size: float = 50,
        probability_threshold: float = 0.3
    ):
        """
        Initialize classifier with detection thresholds.
        
        Args:
            lesion_ratio_threshold: Minimum lesion area ratio to classify as caries
            min_component_size: Minimum connected component size (pixels)
            probability_threshold: Minimum model confidence for detection
        """
        self.lesion_ratio_threshold = lesion_ratio_threshold
        self.min_component_size = min_component_size
        self.probability_threshold = probability_threshold
    
    def classify(self, features: Dict[str, float]) -> Tuple[int, float, str]:
        """
        Classify based on extracted features.
        
        Args:
            features: Dictionary of features from feature_extraction.py
        
        Returns:
            Tuple of (label, confidence, reasoning)
            - label: 1 = CARIES, 0 = NO CARIES
            - confidence: 0.0 to 1.0 classification confidence
            - reasoning: Human-readable explanation
        """
        # Extract relevant features
        lesion_ratio = features.get('lesion_ratio', 0)
        num_components = features.get('num_components', 0)
        max_component_size = features.get('max_component_size', 0)
        max_prob = features.get('max_probability', 0)
        lesion_mean_prob = features.get('lesion_mean_probability', 0)
        
        # Decision logic
        reasons = []
        caries_score = 0.0
        
        # Check lesion area
        if lesion_ratio >= self.lesion_ratio_threshold:
            caries_score += 0.3
            reasons.append(f"Lesion area detected ({lesion_ratio:.4%} of image)")
        
        # Check component size
        if max_component_size >= self.min_component_size:
            caries_score += 0.3
            reasons.append(f"Significant lesion size ({max_component_size:.0f} pixels)")
        
        # Check probability confidence
        if max_prob >= self.probability_threshold:
            caries_score += 0.2
            reasons.append(f"High model confidence ({max_prob:.2%})")
        
        # Check lesion region probability
        if lesion_mean_prob >= self.probability_threshold:
            caries_score += 0.2
            reasons.append(f"Confident lesion region ({lesion_mean_prob:.2%})")
        
        # Final decision
        label = 1 if caries_score >= 0.5 else 0
        confidence = min(caries_score, 1.0)
        
        if label == 1:
            reasoning = "CARIES DETECTED: " + "; ".join(reasons)
        else:
            reasoning = "NO CARIES: No significant lesions detected"
            if reasons:
                reasoning += f" (partial signals: {'; '.join(reasons)})"
        
        return label, confidence, reasoning
    
    def classify_from_mask(
        self,
        mask: np.ndarray,
        probability_map: np.ndarray = None
    ) -> Tuple[int, float, str]:
        """
        Classify directly from mask and probability map.
        """
        from src.classification.feature_extraction import extract_features
        
        features = extract_features(mask, probability_map)
        return self.classify(features)


def get_label_name(label: int) -> str:
    """Convert numeric label to string."""
    return "CARIES" if label == 1 else "NO CARIES"


def print_classification_result(
    label: int,
    confidence: float,
    reasoning: str
) -> None:
    """
    Print classification result in a formatted way.
    """
    label_name = get_label_name(label)
    
    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"  Prediction:   {label_name}")
    print(f"  Confidence:   {confidence:.2%}")
    print(f"  Reasoning:    {reasoning}")
    print("=" * 50)
    
    # Medical disclaimer
    print("\n  [DISCLAIMER] This is a decision-support tool.")
    print("  Final diagnosis must be made by a qualified dentist.")
    print("=" * 50)
