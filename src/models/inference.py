"""
Inference Module for Dental Caries Segmentation
================================================
This module provides inference utilities for running the trained
segmentation model on new images.

Features:
- Load trained model weights
- Run inference on single images or batches
- Post-process predictions
- Save results to disk

Compatible with both VS Code and Google Colab.

Author: Hack4Health Team
"""

import os
import cv2
import numpy as np
import torch

# Import from local modules
from src.models import AttentionUNet
from src.preprocessing import preprocess_single_image


def load_model(model_path, device=None):
    """
    Load a trained Attention U-Net model from disk.
    
    Args:
        model_path (str): Path to the saved model weights (.pth file)
        device (str, optional): Device to load model on ('cuda' or 'cpu')
                               If None, automatically selects based on availability
    
    Returns:
        tuple: (model, device) - Loaded model and device string
    
    Example:
        >>> model, device = load_model("results/models/best_model.pth")
        >>> print(f"Model loaded on {device}")
    """
    # Auto-select device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model architecture
    model = AttentionUNet(in_channels=1, out_channels=1)
    
    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[OK] Model loaded from: {model_path}")
    else:
        print(f"[WARNING] Model file not found: {model_path}")
        print("   Using randomly initialized weights.")
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_for_inference(image, image_size=(256, 256)):
    """
    Preprocess a single image for inference.
    
    Applies the same preprocessing pipeline used during training:
    1. Convert to grayscale (if needed)
    2. Apply CLAHE
    3. Apply Gaussian filtering
    4. Normalize to [0, 1]
    5. Resize to target size
    6. Convert to tensor
    
    Args:
        image (np.ndarray or str): Image array or path to image
        image_size (tuple): Target size (height, width)
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 1, H, W]
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Store original size for later resizing
    original_size = image.shape[:2]
    
    # Apply preprocessing (CLAHE, filtering, normalization)
    processed = preprocess_single_image(image)
    
    # Resize to target size
    processed = cv2.resize(processed, image_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor: [H, W] -> [1, 1, H, W]
    tensor = torch.from_numpy(processed).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor, original_size


def predict_single(model, image, device='cpu', threshold=0.5, 
                   return_probability=False, original_size=None):
    """
    Run inference on a single image.
    
    Args:
        model: Trained segmentation model
        image (torch.Tensor): Preprocessed image tensor [1, 1, H, W]
        device (str): Device to run inference on
        threshold (float): Threshold for binarizing predictions
        return_probability (bool): If True, return probability map instead of binary
        original_size (tuple): If provided, resize output to this size
    
    Returns:
        np.ndarray: Predicted segmentation mask
    """
    # Move to device
    image = image.to(device)
    
    # Run inference (no gradient computation needed)
    with torch.no_grad():
        output = model(image)
    
    # Convert to numpy
    pred = output.squeeze().cpu().numpy()
    
    # Resize to original size if requested
    if original_size is not None:
        pred = cv2.resize(pred, (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    # Return probability or binary mask
    if return_probability:
        return pred
    else:
        return (pred > threshold).astype(np.uint8) * 255


def predict_batch(model, images, device='cpu', threshold=0.5):
    """
    Run inference on a batch of images.
    
    Args:
        model: Trained segmentation model
        images (torch.Tensor): Batch of preprocessed images [B, 1, H, W]
        device (str): Device to run inference on
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        np.ndarray: Batch of predicted masks [B, H, W]
    """
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    # Convert to numpy and binarize
    preds = outputs.squeeze(1).cpu().numpy()
    preds = (preds > threshold).astype(np.uint8) * 255
    
    return preds


def segment_image(image_path, model_path, output_path=None, 
                  threshold=0.5, save_overlay=True):
    """
    High-level function to segment a dental X-ray image.
    
    This is the main entry point for inference from external scripts.
    
    Args:
        image_path (str): Path to input dental X-ray image
        model_path (str): Path to trained model weights
        output_path (str, optional): Path to save segmentation result
        threshold (float): Threshold for binarization
        save_overlay (bool): If True, also save overlay visualization
    
    Returns:
        np.ndarray: Predicted segmentation mask
    
    Example:
        >>> mask = segment_image(
        ...     image_path="data/raw/images/tooth_001.png",
        ...     model_path="results/models/best_model.pth",
        ...     output_path="results/predictions/tooth_001_mask.png"
        ... )
    """
    # Load model
    model, device = load_model(model_path)
    
    # Load and preprocess image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    tensor, original_size = preprocess_for_inference(image_path)
    
    # Run inference
    mask = predict_single(
        model, tensor, device, 
        threshold=threshold, 
        original_size=original_size
    )
    
    # Save results if output path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, mask)
        print(f"[OK] Mask saved to: {output_path}")
        
        # Save overlay if requested
        if save_overlay:
            overlay = create_overlay(original_image, mask)
            overlay_path = output_path.replace('.png', '_overlay.png')
            cv2.imwrite(overlay_path, overlay)
            print(f"[OK] Overlay saved to: {overlay_path}")
    
    return mask


def create_overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    """
    Create an overlay visualization of mask on image.
    
    Args:
        image (np.ndarray): Original grayscale image
        mask (np.ndarray): Binary segmentation mask
        color (tuple): BGR color for the mask overlay
        alpha (float): Transparency of the overlay
    
    Returns:
        np.ndarray: RGB image with mask overlay
    """
    # Convert grayscale to BGR
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    
    # Create colored overlay
    overlay = image_bgr.copy()
    overlay[mask > 127] = color
    
    # Blend
    result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    
    return result


def batch_inference(image_dir, model_path, output_dir, threshold=0.5):
    """
    Run inference on all images in a directory.
    
    Args:
        image_dir (str): Directory containing input images
        model_path (str): Path to trained model weights
        output_dir (str): Directory to save predictions
        threshold (float): Threshold for binarization
    
    Returns:
        int: Number of images processed
    """
    # Load model once
    model, device = load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load and preprocess
            tensor, original_size = preprocess_for_inference(image_path)
            
            # Predict
            mask = predict_single(
                model, tensor, device, 
                threshold=threshold,
                original_size=original_size
            )
            
            # Save
            cv2.imwrite(output_path, mask)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
    
    print(f"[OK] Processed {len(image_files)} images")
    return len(image_files)


# =============================================================================
# MAIN - Example usage
# =============================================================================

if __name__ == "__main__":
    """
    Example inference usage.
    
    Before running, ensure you have:
    1. A trained model saved to results/models/
    2. Test images to segment
    """
    print("=" * 50)
    print("Dental Caries Segmentation - Inference")
    print("=" * 50)
    
    # Example paths (adjust as needed)
    model_path = "results/models/best_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[INFO] Model not found at: {model_path}")
        print("   Training is required before inference.")
        print("   Creating model architecture check only...")
        
        # Test model creation
        model, device = load_model(model_path)
        print(f"   Model created on: {device}")
        print(f"   Parameters: {model.get_num_parameters():,}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 1, 256, 256).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Test input:  {dummy_input.shape}")
        print(f"   Test output: {output.shape}")
        print("\n[OK] Inference pipeline ready!")
    else:
        print(f"[OK] Model found at: {model_path}")
        print("   Ready for inference.")
