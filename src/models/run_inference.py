"""
Dental Caries Segmentation - Inference Script
==============================================
This script performs inference on dental X-ray images using Attention U-Net.

IMPORTANT - Understanding the Visualization:
--------------------------------------------
The model outputs sigmoid probabilities (0.0 to 1.0) for each pixel,
representing the model's CONFIDENCE that a pixel belongs to a caries region.

- Panel 1: Original X-ray input
- Panel 2: Raw probability map (model's direct output)
- Panel 3: Confidence mask (normalized, smoothed probability visualization)
- Panel 4: Heatmap overlay (confidence blended with original image)

MEDICAL DISCLAIMER:
This is a decision-support visualization tool, NOT a diagnostic system.
All outputs represent model confidence/uncertainty and must be reviewed
by qualified dental professionals before any clinical decisions.

Author: Hack4Health Team
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.attention_unet import AttentionUNet


# =============================================================================
# EVALUATION METRICS (use when ground truth mask is available)
# =============================================================================

def dice_score(pred, gt, eps=1e-7):
    """
    Compute Dice coefficient between prediction and ground truth.
    Dice = 2 * |intersection| / (|pred| + |gt|)
    Range: 0.0 (no overlap) to 1.0 (perfect overlap)
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    intersection = np.sum(pred * gt)
    return (2 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)


def iou_score(pred, gt, eps=1e-7):
    """
    Compute IoU (Intersection over Union) between prediction and ground truth.
    IoU = |intersection| / |union|
    Range: 0.0 (no overlap) to 1.0 (perfect overlap)
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + eps) / (union + eps)


def load_model(checkpoint_path, device):
    """Load trained Attention U-Net model."""
    model = AttentionUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=(256, 256)):
    """Load and preprocess dental X-ray image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, 1))
    return torch.from_numpy(image)


def get_input_image():
    """Find the single image in input_image/ folder."""
    input_dir = "input_image"
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Input folder '{input_dir}' not found. "
            "Please create it and place one image inside."
        )
    
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    images = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    if len(images) == 0:
        raise FileNotFoundError(
            f"No images found in '{input_dir}'. "
            "Please place one dental X-ray image inside."
        )
    
    if len(images) > 1:
        print(f"[WARNING] Multiple images found. Using: {images[0]}")
    
    return os.path.join(input_dir, images[0])


def normalize_probability_map(prob_map):
    """
    Normalize probability map using min-max normalization.
    
    This ensures the confidence mask shows meaningful intensity variations
    even when the model's raw output range is small (e.g., 0.0 to 0.5).
    
    Returns:
        Normalized map scaled to 0-255 (uint8)
    """
    min_val = prob_map.min()
    max_val = prob_map.max()
    
    if max_val - min_val < 1e-6:
        # Avoid division by zero if map is uniform
        return np.zeros_like(prob_map, dtype=np.uint8)
    
    # Min-max normalize to [0, 1] then scale to [0, 255]
    normalized = (prob_map - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def create_confidence_mask(prob_map):
    """
    Create confidence mask from raw probability map.
    
    This mask is:
    1. Min-max normalized to use full 0-255 range
    2. Gaussian smoothed for spatial coherence
    
    The resulting mask shows WHERE the model is paying attention,
    with intensity proportional to relative confidence.
    """
    # Normalize to use full dynamic range
    confidence = normalize_probability_map(prob_map)
    
    # Smooth for better visualization (reduces noise, improves coherence)
    confidence = cv2.GaussianBlur(confidence, (5, 5), 0)
    
    return confidence


def create_heatmap_overlay(original_image, confidence_mask, alpha=0.5):
    """
    Create heatmap overlay by blending confidence mask with original X-ray.
    
    Uses INFERNO colormap for medical imaging (dark=low, bright=high confidence).
    """
    # Ensure original is uint8
    if original_image.dtype != np.uint8:
        original_uint8 = (original_image * 255).astype(np.uint8)
    else:
        original_uint8 = original_image.copy()
    
    # Convert grayscale to BGR
    base_image = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2BGR)
    
    # Create colored heatmap from confidence mask
    heatmap = cv2.applyColorMap(confidence_mask, cv2.COLORMAP_INFERNO)
    
    # Alpha blend: overlay = (1-alpha)*base + alpha*heatmap
    overlay = cv2.addWeighted(base_image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def main():
    """Main inference pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    checkpoint_path = "results/checkpoints/best_model.pth"
    OUTPUT_DIR = "results/predictions"
    
    # Get input image
    input_image_path = get_input_image()
    print(f"[INFO] Processing image: {input_image_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path, device)
    print("[INFO] Model loaded successfully")

    # Preprocess image
    image_tensor = preprocess_image(input_image_path).to(device)
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    original_resized = cv2.resize(original_image, (256, 256))

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)

    # Extract raw probability map (sigmoid output, range 0-1)
    prob_map = output.squeeze().cpu().numpy()
    
    # =========================================================================
    # ANALYSIS: Print model prediction statistics
    # =========================================================================
    print(f"\n{'='*50}")
    print("MODEL PREDICTION STATISTICS")
    print(f"{'='*50}")
    print(f"  Raw probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"  Mean probability:      {prob_map.mean():.4f}")
    print(f"  Std deviation:         {prob_map.std():.4f}")
    print(f"{'='*50}\n")
    
    # =========================================================================
    # CREATE CONFIDENCE MASK
    # =========================================================================
    # This is the SINGLE mask panel that shows model confidence
    # Intensity = relative confidence (normalized from raw probabilities)
    confidence_mask = create_confidence_mask(prob_map)
    
    # =========================================================================
    # CREATE HEATMAP OVERLAY
    # =========================================================================
    # Blend confidence visualization with original X-ray
    overlay = create_heatmap_overlay(original_resized, confidence_mask, alpha=0.5)
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    cv2.imwrite(os.path.join(OUTPUT_DIR, "confidence_mask.png"), confidence_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay.png"), overlay)
    
    # Also save colored confidence mask
    confidence_colored = cv2.applyColorMap(confidence_mask, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "confidence_mask_colored.png"), confidence_colored)
    
    # Save raw probability map (for reference)
    prob_map_uint8 = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "probability_map_raw.png"), prob_map_uint8)

    # =========================================================================
    # VISUALIZATION (4 PANELS)
    # =========================================================================
    # This visualization is designed to be honest and defensible:
    # - No artificial boosting
    # - No misleading binary masks
    # - Clear correspondence between probability and confidence
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Panel 1: Original X-ray
    axes[0].imshow(original_resized, cmap="gray")
    axes[0].set_title("Original X-ray", fontsize=12)
    axes[0].axis("off")

    # Panel 2: Raw Probability Map (direct model output)
    # Using 'hot' colormap with adaptive range to show actual distribution
    im = axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=max(0.1, prob_map.max()))
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Probability Map\n(max: {prob_map.max():.3f})", fontsize=12)
    axes[1].axis("off")

    # Panel 3: Confidence Mask (normalized probability visualization)
    # This shows WHERE the model is paying attention with full dynamic range
    axes[2].imshow(confidence_mask, cmap="inferno")
    axes[2].set_title("Confidence Mask\n(normalized)", fontsize=12)
    axes[2].axis("off")

    # Panel 4: Heatmap Overlay (confidence blended with X-ray)
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Heatmap Overlay", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"[OK] Inference complete. Results saved to {OUTPUT_DIR}/")
    print("\nOutput files:")
    print(f"  - confidence_mask.png         : Normalized confidence visualization")
    print(f"  - confidence_mask_colored.png : Colored confidence (INFERNO)")
    print(f"  - overlay.png                 : X-ray with heatmap overlay")
    print(f"  - probability_map_raw.png     : Raw model output (0-255 scaled)")
    print(f"  - visualization.png           : 4-panel comparison figure")


if __name__ == "__main__":
    main()
