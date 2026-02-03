"""
Attention U-Net Model for Dental Caries Segmentation
=====================================================
This module implements the Attention U-Net architecture for
medical image segmentation, specifically designed for dental
caries detection in X-ray images.

Architecture Overview:
- Encoder: Downsampling path with convolutional blocks
- Decoder: Upsampling path with skip connections
- Attention Gates: Focus on relevant regions, suppress noise

Input: [B, 1, 256, 256] - Grayscale dental X-ray
Output: [B, 1, 256, 256] - Binary segmentation mask

Reference:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

Author: Hack4Health Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block used in both encoder and decoder.
    
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    This is the fundamental building block of U-Net.
    Batch normalization helps with training stability.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant features.
    
    The attention gate learns to suppress irrelevant regions
    and highlight salient features useful for segmentation.
    
    This is the key innovation of Attention U-Net over vanilla U-Net.
    """
    
    def __init__(self, gate_channels, feature_channels, intermediate_channels):
        """
        Args:
            gate_channels (int): Channels from the gating signal (decoder)
            feature_channels (int): Channels from the skip connection (encoder)
            intermediate_channels (int): Channels in the intermediate layer
        """
        super(AttentionGate, self).__init__()
        
        # Transform gating signal
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Transform feature map
        self.W_x = nn.Sequential(
            nn.Conv2d(feature_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip_connection):
        """
        Apply attention to skip connection features.
        
        Args:
            gate: Gating signal from decoder (coarser resolution)
            skip_connection: Feature map from encoder (finer resolution)
        
        Returns:
            Attention-weighted skip connection features
        """
        # Transform both signals
        g = self.W_gate(gate)
        x = self.W_x(skip_connection)
        
        # Upsample gating signal to match skip connection size
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Compute attention: additive attention mechanism
        combined = self.relu(g + x)
        attention = self.psi(combined)
        
        # Apply attention to skip connection
        return skip_connection * attention


class EncoderBlock(nn.Module):
    """
    Encoder block: ConvBlock followed by MaxPooling.
    
    Reduces spatial dimensions by 2x while increasing feature depth.
    """
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Convolution (for skip connection)
        features = self.conv(x)
        # Pooling (for next encoder stage)
        pooled = self.pool(features)
        
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block with attention gate.
    
    Structure: Upsample -> Attention Gate -> Concatenate -> ConvBlock
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels (int): Channels from previous decoder stage
            skip_channels (int): Channels from corresponding encoder stage
            out_channels (int): Output channels after this block
        """
        super(DecoderBlock, self).__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        
        # Attention gate
        self.attention = AttentionGate(
            gate_channels=in_channels // 2,
            feature_channels=skip_channels,
            intermediate_channels=skip_channels // 2
        )
        
        # Convolution after concatenation
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x, skip):
        """
        Args:
            x: Feature map from previous decoder stage
            skip: Skip connection from encoder
        
        Returns:
            Decoded feature map
        """
        # Upsample
        x = self.upsample(x)
        
        # Apply attention to skip connection
        skip_attended = self.attention(x, skip)
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip_attended], dim=1)
        
        # Apply convolution
        x = self.conv(x)
        
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net for dental caries segmentation.
    
    This network processes single-channel grayscale dental X-rays
    and outputs a single-channel binary segmentation mask.
    
    Architecture:
        - 4 encoder stages (downsampling)
        - 1 bottleneck
        - 4 decoder stages with attention gates (upsampling)
        - Final 1x1 convolution for segmentation
    
    Input shape: [B, 1, 256, 256]
    Output shape: [B, 1, 256, 256]
    
    Example:
        >>> model = AttentionUNet(in_channels=1, out_channels=1)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([4, 1, 256, 256])
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        """
        Initialize the Attention U-Net.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale)
            out_channels (int): Number of output channels (1 for binary segmentation)
            features (list): Number of features at each encoder stage
        """
        super(AttentionUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, features[0])      # 256 -> 128
        self.encoder2 = EncoderBlock(features[0], features[1])      # 128 -> 64
        self.encoder3 = EncoderBlock(features[1], features[2])      # 64 -> 32
        self.encoder4 = EncoderBlock(features[2], features[3])      # 32 -> 16
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)   # 16 -> 16
        
        # Decoder path with attention
        self.decoder4 = DecoderBlock(features[3] * 2, features[3], features[3])  # 16 -> 32
        self.decoder3 = DecoderBlock(features[3], features[2], features[2])      # 32 -> 64
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])      # 64 -> 128
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])      # 128 -> 256
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [B, 1, H, W]
        
        Returns:
            Segmentation mask of shape [B, 1, H, W]
        """
        # Encoder path (save skip connections)
        skip1, x = self.encoder1(x)   # skip1: [B, 64, 256, 256]
        skip2, x = self.encoder2(x)   # skip2: [B, 128, 128, 128]
        skip3, x = self.encoder3(x)   # skip3: [B, 256, 64, 64]
        skip4, x = self.encoder4(x)   # skip4: [B, 512, 32, 32]
        
        # Bottleneck
        x = self.bottleneck(x)        # [B, 1024, 16, 16]
        
        # Decoder path with attention gates
        x = self.decoder4(x, skip4)   # [B, 512, 32, 32]
        x = self.decoder3(x, skip3)   # [B, 256, 64, 64]
        x = self.decoder2(x, skip2)   # [B, 128, 128, 128]
        x = self.decoder1(x, skip1)   # [B, 64, 256, 256]
        
        # Final 1x1 convolution
        x = self.final_conv(x)        # [B, 1, 256, 256]
        
        # Sigmoid activation for binary segmentation
        x = torch.sigmoid(x)
        
        return x
        def forward_with_features(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        bottleneck = self.bottleneck(x)

        x = self.decoder4(bottleneck, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        x = self.final_conv(x)
        x = torch.sigmoid(x)

        return x, bottleneck

    
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(in_channels=1, out_channels=1, pretrained=False):
    """
    Factory function to create an Attention U-Net model.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        pretrained (bool): If True, load pretrained weights (not implemented)
    
    Returns:
        AttentionUNet: Initialized model
    """
    model = AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")
    
    return model


# =============================================================================
# TESTING - Verify model input/output shapes
# =============================================================================

if __name__ == "__main__":
    """
    Test the model architecture.
    
    Verifies:
    1. Model can be instantiated
    2. Input shape [B, 1, 256, 256] is accepted
    3. Output shape [B, 1, 256, 256] is produced
    4. Output values are in [0, 1] range (sigmoid activation)
    """
    print("=" * 50)
    print("Testing Attention U-Net Architecture")
    print("=" * 50)
    
    # Create model
    model = AttentionUNet(in_channels=1, out_channels=1)
    print(f"Model created successfully")
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test with sample input
    batch_size = 4
    x = torch.randn(batch_size, 1, 256, 256)
    print(f"\nInput shape:  {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Verify shapes
    assert output.shape == (batch_size, 1, 256, 256), "Output shape mismatch!"
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0, 1] range!"
    
    print("\n[OK] All tests passed!")
