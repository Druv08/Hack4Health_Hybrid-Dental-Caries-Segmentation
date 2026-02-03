# Hybrid Dental Caries Segmentation

## Overview
This project implements a hybrid medical image segmentation framework for detecting dental caries in X-ray images.  
The system combines classical image processing techniques with deep learning-based Attention U-Net architecture.

## Hackathon Context
**Hack4Health** – Medical Image Processing Track  
- **Round 1:** Medical Image Segmentation  
- **Round 2:** Medical Image Classification  

## Key Features
- Classical preprocessing (CLAHE, filtering, normalization)
- Attention U-Net based semantic segmentation
- Hybrid Dice + BCE loss
- Post-processing for clinical realism
- Visual explainability

## Project Structure
```
Hybrid-Dental-Caries-Segmentation/
│
├── data/
│   ├── raw/              # Original dental X-ray images
│   ├── processed/        # Preprocessed images
│   └── splits/           # Train/val/test splits
│
├── src/
│   ├── preprocessing/    # Image preprocessing modules
│   ├── dataset/          # PyTorch dataset classes
│   ├── models/           # Neural network architectures
│   ├── training/         # Training loops and utilities
│   ├── evaluation/       # Metrics and evaluation scripts
│   └── postprocessing/   # Post-processing refinement
│
├── notebooks/            # Jupyter notebooks for exploration
│
├── results/
│   ├── predictions/      # Model predictions
│   ├── visualizations/   # Visual outputs
│   └── metrics/          # Evaluation metrics
│
├── demo/                 # Demo scripts and examples
│
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Druv08/Hack4Health_Hybrid-Dental-Caries-Segmentation.git
cd Hack4Health_Hybrid-Dental-Caries-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Quick start example
from src.preprocessing import preprocess_image
from src.models import AttentionUNet

# Load and preprocess image
image = preprocess_image("path/to/dental_xray.png")

# Run segmentation
model = AttentionUNet()
prediction = model.predict(image)
```

## Methodology

### 1. Preprocessing Pipeline
- Grayscale conversion and normalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian denoising
- Edge enhancement

### 2. Deep Learning Model
- **Architecture:** Attention U-Net
- **Loss Function:** Hybrid Dice + Binary Cross-Entropy
- **Optimizer:** Adam with learning rate scheduling

### 3. Post-processing
- Morphological operations
- Connected component analysis
- Contour refinement

## Ethics Statement
⚠️ **Important:** This system is designed as a **decision-support tool** and does not replace professional dental diagnosis. All predictions should be validated by qualified dental professionals.

## Team
Hack4Health Hackathon Participants

## License
This project is for educational and hackathon purposes.
