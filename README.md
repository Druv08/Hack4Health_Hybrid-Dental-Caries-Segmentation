# Hybrid Dental Caries Segmentation

A PyTorch-based medical image segmentation system for detecting dental caries in X-ray images. Combines classical image preprocessing with deep learning for accurate lesion detection.

## Approach

| Stage | Technique |
|-------|-----------|
| Preprocessing | CLAHE, Median Filtering, Normalization |
| Model | Attention U-Net (encoder-decoder with attention gates) |
| Loss | Dice Loss + Binary Cross-Entropy |
| Post-processing | Morphological operations, Small blob removal |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference
Place your dental X-ray in `input_image/` folder, then:
```bash
python -m src.models.run_inference
```

### 3. Train Model
```bash
python -m src.training.train
```

## Project Structure
```
├── src/
│   ├── dataset/          # Data loading and splitting
│   ├── preprocessing/    # CLAHE, filtering, normalization
│   ├── models/           # Attention U-Net architecture
│   ├── training/         # Training loop and losses
│   ├── evaluation/       # Dice, IoU, precision, recall
│   └── postprocessing/   # Mask refinement
├── input_image/          # Place input X-ray here
├── results/
│   ├── checkpoints/      # Saved model weights
│   └── predictions/      # Output masks and overlays
└── requirements.txt
```

## Outputs

After inference, find results in `results/predictions/`:
- `predicted_mask.png` - Binary segmentation mask
- `overlay.png` - X-ray with highlighted caries regions
- `visualization.png` - Side-by-side comparison

## Team
- Druv Mishra
- Nitish Arul

## Medical Disclaimer

This system is a **decision-support tool** for educational purposes only. It does **not** provide medical diagnosis. All outputs must be reviewed and validated by qualified dental professionals before any clinical decisions are made.

---
*Hack4Health 2026 - Medical Image Processing Track*

