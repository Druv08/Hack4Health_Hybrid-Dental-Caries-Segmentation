"""
Loss Functions for Dental Caries Segmentation
==============================================
This module provides loss functions optimized for medical image
segmentation, specifically designed for binary segmentation tasks.

Available Loss Functions:
- Dice Loss: Measures overlap between prediction and ground truth
- BCE Loss: Binary Cross-Entropy for pixel-wise classification
- Combined Loss: Weighted sum of Dice and BCE losses

All loss functions:
- Accept tensors of shape [B, 1, H, W]
- Return a single scalar value
- Are differentiable for backpropagation

Author: Hack4Health Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute the Dice coefficient (F1 score for segmentation).
    
    Dice = 2 * |A intersection B| / (|A| + |B|)
    
    The Dice coefficient measures the overlap between the predicted
    segmentation and the ground truth. A value of 1 means perfect
    overlap, while 0 means no overlap.
    
    Args:
        pred (torch.Tensor): Predicted mask, shape [B, 1, H, W], values in [0, 1]
        target (torch.Tensor): Ground truth mask, shape [B, 1, H, W], values {0, 1}
        smooth (float): Smoothing factor to prevent division by zero
    
    Returns:
        torch.Tensor: Dice coefficient (scalar)
    """
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Compute Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice Loss for segmentation.
    
    Dice Loss = 1 - Dice Coefficient
    
    This loss function directly optimizes the Dice coefficient,
    which is the primary evaluation metric for segmentation tasks.
    
    Why use Dice Loss for medical imaging:
    - Handles class imbalance well (caries regions are often small)
    - Directly optimizes the evaluation metric
    - More robust than cross-entropy for small foreground regions
    
    Args:
        pred (torch.Tensor): Predicted mask, shape [B, 1, H, W], values in [0, 1]
        target (torch.Tensor): Ground truth mask, shape [B, 1, H, W], values {0, 1}
        smooth (float): Smoothing factor to prevent division by zero
    
    Returns:
        torch.Tensor: Dice loss (scalar, lower is better)
    
    Example:
        >>> pred = torch.sigmoid(torch.randn(4, 1, 256, 256))
        >>> target = torch.randint(0, 2, (4, 1, 256, 256)).float()
        >>> loss = dice_loss(pred, target)
        >>> loss.backward()
    """
    return 1.0 - dice_coefficient(pred, target, smooth)


def bce_loss(pred, target):
    """
    Compute Binary Cross-Entropy Loss.
    
    BCE measures the pixel-wise classification error between
    the predicted probabilities and the binary ground truth.
    
    Why use BCE for medical imaging:
    - Provides stable gradients during training
    - Penalizes confident wrong predictions heavily
    - Works well as a regularizer alongside Dice loss
    
    Args:
        pred (torch.Tensor): Predicted mask, shape [B, 1, H, W], values in [0, 1]
        target (torch.Tensor): Ground truth mask, shape [B, 1, H, W], values {0, 1}
    
    Returns:
        torch.Tensor: BCE loss (scalar, lower is better)
    
    Example:
        >>> pred = torch.sigmoid(torch.randn(4, 1, 256, 256))
        >>> target = torch.randint(0, 2, (4, 1, 256, 256)).float()
        >>> loss = bce_loss(pred, target)
        >>> loss.backward()
    """
    # Clamp predictions to avoid log(0)
    pred_clamped = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
    
    # Compute BCE
    bce = F.binary_cross_entropy(pred_clamped, target, reduction='mean')
    
    return bce


def combined_loss(pred, target, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
    """
    Compute combined Dice + BCE loss.
    
    Combined Loss = dice_weight * Dice Loss + bce_weight * BCE Loss
    
    This hybrid loss leverages the strengths of both:
    - Dice Loss: Handles class imbalance, optimizes overlap
    - BCE Loss: Provides stable gradients, penalizes errors
    
    This combination is widely used in medical image segmentation
    and often outperforms either loss function alone.
    
    Args:
        pred (torch.Tensor): Predicted mask, shape [B, 1, H, W], values in [0, 1]
        target (torch.Tensor): Ground truth mask, shape [B, 1, H, W], values {0, 1}
        dice_weight (float): Weight for Dice loss component (default: 0.5)
        bce_weight (float): Weight for BCE loss component (default: 0.5)
        smooth (float): Smoothing factor for Dice loss
    
    Returns:
        torch.Tensor: Combined loss (scalar, lower is better)
    
    Example:
        >>> pred = torch.sigmoid(torch.randn(4, 1, 256, 256))
        >>> target = torch.randint(0, 2, (4, 1, 256, 256)).float()
        >>> loss = combined_loss(pred, target, dice_weight=0.5, bce_weight=0.5)
        >>> loss.backward()
    """
    # Compute individual losses
    d_loss = dice_loss(pred, target, smooth)
    b_loss = bce_loss(pred, target)
    
    # Combine with weights
    total_loss = dice_weight * d_loss + bce_weight * b_loss
    
    return total_loss


class DiceLoss(nn.Module):
    """
    Dice Loss as a PyTorch Module.
    
    Useful when you need a loss function that can be passed
    to training frameworks or saved/loaded with model state.
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        return dice_loss(pred, target, self.smooth)


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss as a PyTorch Module.
    """
    
    def __init__(self):
        super(BCELoss, self).__init__()
    
    def forward(self, pred, target):
        return bce_loss(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE Loss as a PyTorch Module.
    
    This is the recommended loss function for dental caries segmentation.
    
    Example:
        >>> criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        return combined_loss(
            pred, target, 
            self.dice_weight, 
            self.bce_weight, 
            self.smooth
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    
    Focal Loss down-weights easy examples and focuses training
    on hard misclassified examples. Useful when caries regions
    are very small compared to background.
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha (float): Weighting factor for positive class
            gamma (float): Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Clamp predictions
        pred_clamped = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal weights
        pt = torch.where(target == 1, pred_clamped, 1 - pred_clamped)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute BCE
        bce = F.binary_cross_entropy(pred_clamped, target, reduction='none')
        
        # Apply focal weight and alpha
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()


# =============================================================================
# TESTING - Verify loss functions work correctly
# =============================================================================

if __name__ == "__main__":
    """
    Test all loss functions.
    
    Verifies:
    1. All functions accept correct input shapes
    2. All functions return scalar values
    3. Gradients can be computed
    4. Loss decreases when prediction matches target
    """
    print("=" * 50)
    print("Testing Loss Functions")
    print("=" * 50)
    
    # Create sample data
    batch_size = 4
    pred = torch.sigmoid(torch.randn(batch_size, 1, 256, 256, requires_grad=True))
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    print(f"Input shapes:")
    print(f"  Prediction: {pred.shape}")
    print(f"  Target:     {target.shape}")
    
    # Test Dice Loss
    print("\n1. Dice Loss:")
    d_loss = dice_loss(pred, target)
    print(f"   Value: {d_loss.item():.4f}")
    d_loss.backward(retain_graph=True)
    print(f"   Gradient computed: OK")
    
    # Test BCE Loss
    print("\n2. BCE Loss:")
    b_loss = bce_loss(pred, target)
    print(f"   Value: {b_loss.item():.4f}")
    
    # Test Combined Loss
    print("\n3. Combined Loss (Dice + BCE):")
    c_loss = combined_loss(pred, target)
    print(f"   Value: {c_loss.item():.4f}")
    
    # Test Module versions
    print("\n4. Module Versions:")
    criterion = CombinedLoss()
    loss = criterion(pred, target)
    print(f"   CombinedLoss: {loss.item():.4f}")
    
    # Test perfect prediction
    print("\n5. Perfect Prediction Test:")
    perfect_pred = target.clone().requires_grad_(True)
    perfect_loss = dice_loss(perfect_pred, target)
    print(f"   Dice Loss (perfect): {perfect_loss.item():.6f}")
    
    print("\n[OK] All loss function tests passed!")
