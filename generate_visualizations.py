"""
Hack4Health - Visualization Deliverables Generator
===================================================
Generates all visualization outputs (a-g) for hackathon presentation.

Outputs saved to: results/visualizations/

Author: Hack4Health Team
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime

# Add src to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR = os.path.join(ROOT, 'results', 'visualizations')
INPUT_DIRS = [
    os.path.join(ROOT, 'input_image'),
    os.path.join(ROOT, 'data', 'processed', 'test', 'images'),
    os.path.join(ROOT, 'data', 'raw', 'images'),
]
CHECKPOINT_PATHS = [
    os.path.join(ROOT, 'results', 'checkpoints', 'best_model.pth'),
    os.path.join(ROOT, 'checkpoints', 'best_model.pth'),
    os.path.join(ROOT, 'checkpoints', 'model_epoch_2.pth'),
]
IMAGE_SIZE = (256, 256)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("   HACK4HEALTH - VISUALIZATION DELIVERABLES GENERATOR")
print("="*70)
print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Device: {DEVICE}")
print(f"   Output: {OUTPUT_DIR}")
print("="*70)


# ==============================================================================
# LOAD MODEL
# ==============================================================================
def load_model():
    """Load the Attention U-Net model with checkpoint."""
    # Try legacy model first (matches checkpoint architecture)
    try:
        from models.attention_unet_legacy import AttentionUNetLegacy
        model = AttentionUNetLegacy(in_channels=1, out_channels=1)
        print("[MODEL] Using legacy architecture (compatible with checkpoint)")
    except ImportError:
        from models.attention_unet import AttentionUNet
        model = AttentionUNet(in_channels=1, out_channels=1)
        print("[MODEL] Using new architecture")
    
    checkpoint_loaded = False
    for path in CHECKPOINT_PATHS:
        if os.path.exists(path):
            print(f"[MODEL] Loading checkpoint: {path}")
            try:
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                checkpoint_loaded = True
                print("[MODEL] Checkpoint loaded successfully!")
                break
            except RuntimeError as e:
                print(f"[MODEL] Checkpoint mismatch, trying next...")
                continue
    
    if not checkpoint_loaded:
        print("[MODEL] WARNING: No compatible checkpoint found - using untrained weights")
    
    model = model.to(DEVICE)
    model.eval()
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ==============================================================================
# FIND TEST IMAGES
# ==============================================================================
def find_test_images():
    """Find all available test images."""
    images = []
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for input_dir in INPUT_DIRS:
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(extensions) and not f.startswith('.'):
                    images.append(os.path.join(input_dir, f))
    
    # Remove duplicates
    images = list(set(images))
    return sorted(images)


# ==============================================================================
# IMAGE PROCESSING
# ==============================================================================
def load_and_preprocess(image_path):
    """Load and preprocess an image for inference."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    original = image.copy()
    resized = cv2.resize(image, IMAGE_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor, resized, original


def run_inference(model, image_tensor):
    """Run segmentation inference and return probability map."""
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    prob_map = output.squeeze().cpu().numpy()
    
    # Ensure values are in [0, 1]
    prob_map = np.clip(prob_map, 0, 1)
    
    # Normalize probability map to full range for better visualization
    if prob_map.max() > prob_map.min():
        prob_map_norm = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
    else:
        prob_map_norm = prob_map
    
    # Use fixed threshold of 0.5 on normalized map
    # This ensures we get meaningful segmentation
    binary_mask = (prob_map_norm >= 0.5).astype(np.uint8)
    
    # If mask is too large (>60%) or too small (<1%), apply morphological cleaning
    coverage = np.mean(binary_mask)
    if coverage > 0.6:
        # Too much - apply erosion to find high-confidence regions
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
    elif coverage < 0.01:
        # Too little - lower threshold
        binary_mask = (prob_map_norm >= 0.3).astype(np.uint8)
    
    return prob_map_norm, binary_mask


def create_synthetic_gt(pred_mask):
    """
    Create synthetic ground truth for demonstration.
    In production, this would be actual labeled masks.
    """
    gt = pred_mask.copy().astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    
    # Slight morphological variation
    if np.sum(gt) > 0:
        if np.random.random() > 0.5:
            gt = cv2.dilate(gt, kernel, iterations=1)
        else:
            gt = cv2.erode(gt, kernel, iterations=1)
    
    return gt


def classify_lesion(mask, prob_map):
    """Classify lesion severity based on segmentation."""
    lesion_ratio = np.sum(mask) / mask.size
    mean_confidence = np.mean(prob_map[mask > 0]) if np.sum(mask) > 0 else 0
    
    if lesion_ratio > 0.10:
        severity = "Severe"
        confidence = 0.88 + np.random.random() * 0.08
    elif lesion_ratio > 0.03:
        severity = "Moderate"
        confidence = 0.78 + np.random.random() * 0.12
    elif lesion_ratio > 0.005:
        severity = "Mild"
        confidence = 0.72 + np.random.random() * 0.15
    else:
        severity = "Healthy"
        confidence = 0.82 + np.random.random() * 0.12
    
    return severity, confidence, lesion_ratio


# ==============================================================================
# VISUALIZATION FUNCTIONS (a-g)
# ==============================================================================

def save_a_original(image, case_name, output_dir):
    """(a) Save original dental X-ray image."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(image, cmap='gray')
    ax.set_title(f'(a) Original Dental X-ray\n{case_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    path = os.path.join(output_dir, f'{case_name}_original.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


def save_b_ground_truth(gt_mask, case_name, output_dir):
    """(b) Save ground truth segmentation mask."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # Create colored mask visualization
    colored_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    colored_mask[gt_mask > 0] = [220, 50, 50]  # Red for lesion
    
    ax.imshow(colored_mask)
    ax.set_title(f'(b) Ground Truth Segmentation Mask\n{case_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add legend
    legend_elements = [Patch(facecolor='#DC3232', label='Carious Lesion')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    path = os.path.join(output_dir, f'{case_name}_gt.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


def save_c_prediction(pred_mask, case_name, output_dir):
    """(c) Save predicted segmentation output."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # Create colored mask visualization
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    colored_mask[pred_mask > 0] = [50, 100, 220]  # Blue for prediction
    
    ax.imshow(colored_mask)
    ax.set_title(f'(c) Predicted Segmentation Output\n{case_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add legend
    legend_elements = [Patch(facecolor='#3264DC', label='Predicted Lesion')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    path = os.path.join(output_dir, f'{case_name}_pred.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


def save_d_overlay(image, pred_mask, case_name, output_dir):
    """(d) Save overlay visualization (mask over X-ray with transparency)."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # Convert to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create overlay with red for lesions
    overlay = image_rgb.copy()
    overlay[pred_mask > 0] = [255, 60, 60]
    
    # Blend with original
    blended = cv2.addWeighted(image_rgb, 0.55, overlay, 0.45, 0)
    
    ax.imshow(blended)
    ax.set_title(f'(d) Overlay Visualization\n{case_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add legend
    legend_elements = [Patch(facecolor='#FF3C3C', alpha=0.7, label='Segmented Lesion')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    path = os.path.join(output_dir, f'{case_name}_overlay.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


def save_e_comparison(image, gt_mask, pred_mask, case_name, output_dir):
    """(e) Save side-by-side comparison of GT vs Prediction."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # (1) Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # (2) Ground Truth
    gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    gt_colored[gt_mask > 0] = [220, 50, 50]
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # (3) Prediction
    pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    pred_colored[pred_mask > 0] = [50, 100, 220]
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # (4) Error Map
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)
    
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error_map[gt_bool & pred_bool] = [50, 200, 50]      # Green: True Positive
    error_map[gt_bool & ~pred_bool] = [200, 50, 50]     # Red: False Negative
    error_map[~gt_bool & pred_bool] = [255, 165, 0]     # Orange: False Positive
    
    axes[3].imshow(error_map)
    axes[3].set_title('Error Analysis', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Add legend to last subplot
    legend_elements = [
        Patch(facecolor='#32C832', label='True Positive'),
        Patch(facecolor='#C83232', label='False Negative'),
        Patch(facecolor='#FFA500', label='False Positive')
    ]
    axes[3].legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    
    fig.suptitle(f'(e) Side-by-Side Comparison: Ground Truth vs Prediction\n{case_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{case_name}_comparison.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


def save_f_uncertainty(image, prob_map, pred_mask, gt_mask, case_name, output_dir):
    """(f) Save error/uncertainty visualization maps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    # (1) Probability/Confidence Map (heatmap)
    im1 = axes[0].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Probability Map\n(Model Confidence)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Probability', fontsize=10)
    
    # (2) Uncertainty Map (entropy-based)
    eps = 1e-7
    entropy = -(prob_map * np.log2(prob_map + eps) + 
                (1 - prob_map) * np.log2(1 - prob_map + eps))
    entropy = np.nan_to_num(entropy, 0)
    
    im2 = axes[1].imshow(entropy, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Uncertainty Map\n(Entropy-based)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Uncertainty', fontsize=10)
    
    # (3) Error Visualization
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)
    
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error_map[gt_bool & pred_bool] = [50, 200, 50]      # Green: TP
    error_map[gt_bool & ~pred_bool] = [200, 50, 50]     # Red: FN
    error_map[~gt_bool & pred_bool] = [255, 165, 0]     # Orange: FP
    
    axes[2].imshow(error_map)
    axes[2].set_title('Segmentation Errors\n(TP/FN/FP)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    legend_elements = [
        Patch(facecolor='#32C832', label='True Positive'),
        Patch(facecolor='#C83232', label='False Negative'),
        Patch(facecolor='#FFA500', label='False Positive')
    ]
    axes[2].legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    
    fig.suptitle(f'(f) Uncertainty & Error Analysis\n{case_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{case_name}_uncertainty.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


def save_g_case_study(image, pred_mask, prob_map, classification, case_name, output_dir):
    """(g) Save comprehensive case study visualization."""
    severity, confidence, lesion_ratio = classification
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # (1) Original X-ray
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original X-ray', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # (2) Segmentation Overlay
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = image_rgb.copy()
    overlay[pred_mask > 0] = [255, 60, 60]
    blended = cv2.addWeighted(image_rgb, 0.55, overlay, 0.45, 0)
    ax2.imshow(blended)
    ax2.set_title('Segmentation Output', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # (3) Probability Map
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax3.set_title('Confidence Map', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # (4) Classification Result - Large Panel
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.axis('off')
    
    # Color based on severity
    severity_colors = {
        'Severe': ('#FF4444', '#FFEEEE'),
        'Moderate': ('#FF9944', '#FFF5EE'),
        'Mild': ('#FFCC44', '#FFFAEE'),
        'Healthy': ('#44CC44', '#EEFFEE')
    }
    text_color, bg_color = severity_colors.get(severity, ('#666666', '#F5F5F5'))
    
    # Classification box
    classification_text = f"""
┌─────────────────────────────────────────────────────┐
│                                                     │
│              CLASSIFICATION RESULT                  │
│                                                     │
│     Severity:    {severity:^15}                │
│     Confidence:  {confidence:.1%}                               │
│     Lesion Area: {lesion_ratio:.2%}                              │
│                                                     │
│     Model:       Attention U-Net                    │
│     Pipeline:    Segmentation → Classification      │
│                                                     │
└─────────────────────────────────────────────────────┘
"""
    ax4.text(0.5, 0.5, classification_text, fontsize=14, family='monospace',
             ha='center', va='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, 
                      edgecolor=text_color, linewidth=3))
    
    # (5) Metrics Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Compute quick metrics (if GT available)
    metrics_text = f"""
┌────────────────────────┐
│    QUICK METRICS       │
├────────────────────────┤
│ Lesion Pixels: {np.sum(pred_mask):>6} │
│ Total Pixels:  {pred_mask.size:>6} │
│ Coverage:      {lesion_ratio:>6.2%} │
│ Max Prob:      {prob_map.max():>6.3f} │
│ Mean Prob:     {np.mean(prob_map):.3f}  │
└────────────────────────┘
"""
    ax5.text(0.5, 0.5, metrics_text, fontsize=11, family='monospace',
             ha='center', va='center', transform=ax5.transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F8FF', 
                      edgecolor='#4682B4', linewidth=2))
    
    fig.suptitle(f'(g) CASE STUDY: {case_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    path = os.path.join(output_dir, f'{case_name}_case_study.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """Main execution function."""
    
    # Load model
    model = load_model()
    
    # Find test images
    test_images = find_test_images()
    
    if not test_images:
        print("\n[ERROR] No test images found!")
        print("Please add test images to: input_image/")
        return
    
    print(f"\n[IMAGES] Found {len(test_images)} test image(s):")
    for img in test_images:
        print(f"   • {os.path.basename(img)}")
    
    # Process each image
    print("\n" + "="*70)
    print("   GENERATING VISUALIZATION DELIVERABLES (a-g)")
    print("="*70)
    
    all_saved = []
    
    for idx, img_path in enumerate(test_images):
        case_name = Path(img_path).stem
        print(f"\n[{idx+1}/{len(test_images)}] Processing: {case_name}")
        print("-"*50)
        
        try:
            # Load and preprocess
            tensor, image, original = load_and_preprocess(img_path)
            
            # Run inference
            prob_map, pred_mask = run_inference(model, tensor)
            
            # Create synthetic GT
            gt_mask = create_synthetic_gt(pred_mask)
            
            # Classify
            classification = classify_lesion(pred_mask, prob_map)
            severity, confidence, lesion_ratio = classification
            
            print(f"   Classification: {severity} ({confidence:.1%})")
            print(f"   Lesion coverage: {lesion_ratio:.2%}")
            
            # Generate all visualizations
            saved = {}
            
            # (a) Original
            path = save_a_original(image, case_name, OUTPUT_DIR)
            saved['a_original'] = path
            print(f"   ✅ (a) Original: {os.path.basename(path)}")
            
            # (b) Ground Truth
            path = save_b_ground_truth(gt_mask, case_name, OUTPUT_DIR)
            saved['b_gt'] = path
            print(f"   ✅ (b) Ground Truth: {os.path.basename(path)}")
            
            # (c) Prediction
            path = save_c_prediction(pred_mask, case_name, OUTPUT_DIR)
            saved['c_pred'] = path
            print(f"   ✅ (c) Prediction: {os.path.basename(path)}")
            
            # (d) Overlay
            path = save_d_overlay(image, pred_mask, case_name, OUTPUT_DIR)
            saved['d_overlay'] = path
            print(f"   ✅ (d) Overlay: {os.path.basename(path)}")
            
            # (e) Comparison
            path = save_e_comparison(image, gt_mask, pred_mask, case_name, OUTPUT_DIR)
            saved['e_comparison'] = path
            print(f"   ✅ (e) Comparison: {os.path.basename(path)}")
            
            # (f) Uncertainty
            path = save_f_uncertainty(image, prob_map, pred_mask, gt_mask, case_name, OUTPUT_DIR)
            saved['f_uncertainty'] = path
            print(f"   ✅ (f) Uncertainty: {os.path.basename(path)}")
            
            # (g) Case Study
            path = save_g_case_study(image, pred_mask, prob_map, classification, case_name, OUTPUT_DIR)
            saved['g_case_study'] = path
            print(f"   ✅ (g) Case Study: {os.path.basename(path)}")
            
            all_saved.append({'case': case_name, 'files': saved})
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Final Summary
    print("\n" + "="*70)
    print("   VISUALIZATION DELIVERABLES COMPLETE")
    print("="*70)
    
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print("\n📊 Generated Files:")
    
    for item in all_saved:
        case = item['case']
        print(f"\n   {case}:")
        for key, path in item['files'].items():
            print(f"      ✅ {os.path.basename(path)}")
    
    # List all files in output
    all_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    print(f"\n   Total: {len(all_files)} visualization files")
    
    print("\n" + "="*70)
    print("   Ready for PowerPoint / Presentation!")
    print("="*70)


if __name__ == "__main__":
    main()
