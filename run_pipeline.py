"""
Hybrid Dental Caries Detection Pipeline
========================================
Round 2: Complete End-to-End Pipeline

Workflow:
1. Load dental X-ray image
2. Run segmentation (Attention U-Net)
3. Apply post-processing (morphology)
4. Extract features from mask
5. Classify as caries/no-caries
6. Generate visualization and report

Usage:
    python run_pipeline.py --image input_image/benign-2.png
    python run_pipeline.py --image path/to/image.png --output results/
    python run_pipeline.py --batch input_image/ --output batch_results/

Author: Hack4Health Team
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline components
from models.attention_unet import AttentionUNet
from postprocessing.refine_mask import (
    postprocess_mask,
    remove_small_components,
    smooth_mask
)
from classification.feature_extraction import extract_features, print_features
from classification.classifier import CariesClassifier, print_classification_result


def load_model(checkpoint_path: str = None, device: str = 'cuda') -> AttentionUNet:
    """Load the segmentation model."""
    model = AttentionUNet(in_channels=1, out_channels=1)
    
    # Try to find checkpoint
    if checkpoint_path is None:
        # Default checkpoint locations
        possible_paths = [
            'checkpoints/best_model.pth',
            'checkpoints/model_epoch_2.pth',
            'checkpoints/model_final.pth',
            'model.pth'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("[INFO] Checkpoint loaded successfully")
    else:
        print("[WARNING] No checkpoint found - using random weights")
    
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, target_size: tuple = (256, 256)) -> tuple:
    """
    Preprocess image for inference.
    
    Returns:
        (tensor, original_image, original_size)
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    original_size = image.shape[:2]
    original_image = image.copy()
    
    # Resize
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor, original_image, original_size


def run_segmentation(model: AttentionUNet, image_tensor: torch.Tensor, 
                     device: str = 'cuda', threshold: float = 0.5) -> tuple:
    """
    Run segmentation inference.
    
    Returns:
        (binary_mask, probability_map)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = torch.sigmoid(output)
    
    # Convert to numpy
    prob_map_np = prob_map.squeeze().cpu().numpy()
    
    # Normalize probability map if needed
    if prob_map_np.max() > prob_map_np.min():
        prob_map_np = (prob_map_np - prob_map_np.min()) / (prob_map_np.max() - prob_map_np.min())
    
    # Threshold
    binary_mask = (prob_map_np >= threshold).astype(np.uint8) * 255
    
    return binary_mask, prob_map_np


def run_postprocessing(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Apply post-processing to refine the mask."""
    # Use comprehensive refinement
    refined = postprocess_mask(mask)
    
    return refined


def run_classification(mask: np.ndarray, verbose: bool = True) -> dict:
    """
    Run feature extraction and classification.
    
    Returns:
        Dictionary with features, label, confidence, reasoning
    """
    # Extract features
    features = extract_features(mask)
    
    if verbose:
        print("\n" + "="*50)
        print("FEATURE EXTRACTION RESULTS")
        print("="*50)
        print_features(features)
    
    # Classify
    classifier = CariesClassifier()
    label, confidence, reasoning = classifier.classify(features)
    
    if verbose:
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        print_classification_result(label, confidence, reasoning)
    
    return {
        'features': features,
        'label': label,
        'label_name': 'CARIES' if label == 1 else 'NO CARIES',
        'confidence': confidence,
        'reasoning': reasoning
    }


def create_visualization(original_image: np.ndarray, 
                        binary_mask: np.ndarray,
                        refined_mask: np.ndarray,
                        prob_map: np.ndarray,
                        classification_result: dict,
                        original_size: tuple) -> np.ndarray:
    """Create a comprehensive visualization panel."""
    h, w = original_size
    target_h, target_w = 256, 256
    
    # Resize masks to match visualization size
    mask_resized = cv2.resize(binary_mask, (target_w, target_h), 
                              interpolation=cv2.INTER_NEAREST)
    refined_resized = cv2.resize(refined_mask, (target_w, target_h), 
                                 interpolation=cv2.INTER_NEAREST)
    prob_resized = cv2.resize(prob_map, (target_w, target_h), 
                              interpolation=cv2.INTER_LINEAR)
    
    # Resize original for display
    original_resized = cv2.resize(original_image, (target_w, target_h))
    
    # Convert to RGB for visualization
    original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
    
    # Create overlay
    overlay = original_rgb.copy()
    mask_colored = np.zeros_like(original_rgb)
    
    # Color caries regions (red for caries, green for no caries)
    if classification_result['label'] == 1:  # Caries
        mask_colored[:, :, 2] = refined_resized  # Red channel
    else:  # No caries
        mask_colored[:, :, 1] = refined_resized  # Green channel
    
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # Create heatmap from probability
    prob_uint8 = (prob_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    
    # Create 4-panel visualization
    panel_h, panel_w = target_h, target_w
    
    # Top row: Original | Overlay
    top_row = np.hstack([original_rgb, overlay])
    
    # Bottom row: Probability Heatmap | Refined Mask
    refined_rgb = cv2.cvtColor(refined_resized, cv2.COLOR_GRAY2BGR)
    bottom_row = np.hstack([heatmap, refined_rgb])
    
    # Combine
    visualization = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Add panel labels
    cv2.putText(visualization, "Original", (10, 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(visualization, "Overlay", (panel_w + 10, 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(visualization, "Probability Heatmap", (10, panel_h + 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(visualization, "Refined Mask", (panel_w + 10, panel_h + 20), font, font_scale, (255, 255, 255), thickness)
    
    # Add classification result
    result_text = f"{classification_result['label_name']} (Conf: {classification_result['confidence']:.2f})"
    cv2.putText(visualization, result_text, (10, 2*panel_h - 10), 
                font, 0.6, (0, 255, 255), 2)
    
    return visualization


def save_results(output_dir: str, image_name: str, 
                 visualization: np.ndarray,
                 binary_mask: np.ndarray,
                 refined_mask: np.ndarray,
                 classification_result: dict) -> dict:
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(image_name).stem
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_pipeline_result.png")
    cv2.imwrite(vis_path, visualization)
    
    # Save masks
    mask_path = os.path.join(output_dir, f"{base_name}_binary_mask.png")
    cv2.imwrite(mask_path, binary_mask)
    
    refined_path = os.path.join(output_dir, f"{base_name}_refined_mask.png")
    cv2.imwrite(refined_path, refined_mask)
    
    # Save report
    report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DENTAL CARIES DETECTION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Image: {image_name}\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-"*60 + "\n")
        f.write("\nCLASSIFICATION RESULT:\n")
        f.write(f"  Label: {classification_result['label_name']}\n")
        f.write(f"  Confidence: {classification_result['confidence']:.4f}\n")
        f.write(f"  Reasoning: {classification_result['reasoning']}\n")
        f.write("\nEXTRACTED FEATURES:\n")
        for key, value in classification_result['features'].items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("="*60 + "\n")
    
    return {
        'visualization': vis_path,
        'binary_mask': mask_path,
        'refined_mask': refined_path,
        'report': report_path
    }


def run_pipeline(image_path: str, 
                 output_dir: str = 'pipeline_output',
                 checkpoint_path: str = None,
                 device: str = None,
                 threshold: float = 0.5,
                 verbose: bool = True) -> dict:
    """
    Run the complete dental caries detection pipeline.
    
    Args:
        image_path: Path to input dental X-ray image
        output_dir: Directory to save results
        checkpoint_path: Path to model checkpoint
        device: Device to use (cuda/cpu)
        threshold: Segmentation threshold
        verbose: Print progress information
        
    Returns:
        Dictionary with all results
    """
    start_time = time.time()
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("\n" + "="*60)
        print("HYBRID DENTAL CARIES DETECTION PIPELINE")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Device: {device}")
        print("-"*60)
    
    # Step 1: Load model
    if verbose:
        print("\n[STEP 1/5] Loading segmentation model...")
    model = load_model(checkpoint_path, device)
    
    # Step 2: Preprocess image
    if verbose:
        print("[STEP 2/5] Preprocessing image...")
    image_tensor, original_image, original_size = preprocess_image(image_path)
    
    # Step 3: Run segmentation
    if verbose:
        print("[STEP 3/5] Running segmentation...")
    binary_mask, prob_map = run_segmentation(model, image_tensor, device, threshold)
    
    # Step 4: Post-processing
    if verbose:
        print("[STEP 4/5] Applying post-processing...")
    refined_mask = run_postprocessing(binary_mask)
    
    # Step 5: Classification
    if verbose:
        print("[STEP 5/5] Running classification...")
    classification_result = run_classification(refined_mask, verbose=verbose)
    
    # Create visualization
    visualization = create_visualization(
        original_image, binary_mask, refined_mask, prob_map,
        classification_result, original_size
    )
    
    # Save results
    saved_paths = save_results(
        output_dir, os.path.basename(image_path),
        visualization, binary_mask, refined_mask, classification_result
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Results saved to: {output_dir}/")
        for key, path in saved_paths.items():
            print(f"  - {key}: {os.path.basename(path)}")
        print("="*60)
    
    return {
        'image_path': image_path,
        'classification': classification_result,
        'saved_paths': saved_paths,
        'elapsed_time': elapsed_time
    }


def run_batch(input_dir: str, output_dir: str = 'batch_results', **kwargs) -> list:
    """Run pipeline on all images in a directory."""
    results = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    image_files = [f for f in os.listdir(input_dir) 
                   if Path(f).suffix.lower() in image_extensions]
    
    print(f"\n[BATCH MODE] Found {len(image_files)} images in {input_dir}")
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        image_path = os.path.join(input_dir, image_file)
        
        try:
            result = run_pipeline(image_path, output_dir, verbose=False, **kwargs)
            results.append(result)
            print(f"  Result: {result['classification']['label_name']} "
                  f"(Confidence: {result['classification']['confidence']:.2f})")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({'image_path': image_path, 'error': str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    successful = [r for r in results if 'error' not in r]
    caries_count = sum(1 for r in successful if r['classification']['label'] == 1)
    no_caries_count = len(successful) - caries_count
    print(f"Total processed: {len(image_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Caries detected: {caries_count}")
    print(f"No caries: {no_caries_count}")
    print(f"Errors: {len(results) - len(successful)}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Dental Caries Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --image input_image/benign-2.png
  python run_pipeline.py --image test.png --output results/
  python run_pipeline.py --batch input_image/ --output batch_results/
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--batch', type=str, help='Directory for batch processing')
    parser.add_argument('--output', type=str, default='pipeline_output',
                        help='Output directory (default: pipeline_output)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Segmentation threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto-detect)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate input
    if args.image is None and args.batch is None:
        # Default: run on first image in input_image/
        if os.path.exists('input_image'):
            images = [f for f in os.listdir('input_image') 
                     if Path(f).suffix.lower() in ['.png', '.jpg', '.jpeg']]
            if images:
                args.image = os.path.join('input_image', images[0])
                print(f"[INFO] No input specified, using: {args.image}")
            else:
                parser.error("No images found in input_image/")
        else:
            parser.error("Please specify --image or --batch")
    
    if args.image and args.batch:
        parser.error("Cannot use both --image and --batch")
    
    # Run pipeline
    if args.batch:
        if not os.path.isdir(args.batch):
            parser.error(f"Batch directory not found: {args.batch}")
        run_batch(args.batch, args.output, 
                  checkpoint_path=args.checkpoint,
                  device=args.device,
                  threshold=args.threshold)
    else:
        if not os.path.exists(args.image):
            parser.error(f"Image not found: {args.image}")
        run_pipeline(args.image, args.output,
                     checkpoint_path=args.checkpoint,
                     device=args.device,
                     threshold=args.threshold,
                     verbose=not args.quiet)


if __name__ == '__main__':
    main()
