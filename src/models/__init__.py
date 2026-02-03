"""
Models Module for Dental Caries Segmentation
=============================================
Contains neural network architectures for medical image segmentation.

Available Models:
- AttentionUNet: U-Net with attention gates for focused segmentation

Inference:
- load_model: Load trained model from disk
- segment_image: High-level inference function
- predict_single: Run inference on single image
"""

from .attention_unet import (
    AttentionUNet,
    ConvBlock,
    AttentionGate,
    EncoderBlock,
    DecoderBlock,
    create_model
)

from .inference import (
    load_model,
    preprocess_for_inference,
    predict_single,
    predict_batch,
    segment_image,
    create_overlay,
    batch_inference
)

__all__ = [
    # Model architecture
    'AttentionUNet',
    'ConvBlock',
    'AttentionGate',
    'EncoderBlock',
    'DecoderBlock',
    'create_model',
    # Inference
    'load_model',
    'preprocess_for_inference',
    'predict_single',
    'predict_batch',
    'segment_image',
    'create_overlay',
    'batch_inference'
]
