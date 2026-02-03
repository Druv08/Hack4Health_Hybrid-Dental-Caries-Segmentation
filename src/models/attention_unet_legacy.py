"""
Legacy Attention U-Net Model - Compatible with existing checkpoint
===================================================================
This version matches the architecture of the trained checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block - NO bias (matches checkpoint)"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections - NO bias (matches checkpoint)"""
    def __init__(self, gate_channels, feature_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(feature_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        x = self.W_x(skip)
        # Resize gate to match skip size
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g + x)
        attention = self.psi(psi)
        return skip * attention


class EncoderBlock(nn.Module):
    """Encoder block with conv and pooling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block with attention - upsample halves channels"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Upsample: in_channels -> out_channels (halves the channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Attention: gate=out_channels, skip=skip_channels
        self.attention = AttentionGate(out_channels, skip_channels, skip_channels // 2)
        # Conv: (out_channels + skip_channels) -> out_channels
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        skip_att = self.attention(x, skip)
        # Handle size mismatch
        if x.shape[2:] != skip_att.shape[2:]:
            x = F.interpolate(x, size=skip_att.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_att], dim=1)
        return self.conv(x)


class AttentionUNetLegacy(nn.Module):
    """
    Attention U-Net with architecture matching the trained checkpoint.
    Uses encoder1-4, decoder1-4 naming convention.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNetLegacy, self).__init__()
        
        # Encoder
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        
        # Decoder
        self.decoder4 = DecoderBlock(features[3] * 2, features[3], features[3])
        self.decoder3 = DecoderBlock(features[3], features[2], features[2])
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with attention
        x = self.decoder4(x, s4)
        x = self.decoder3(x, s3)
        x = self.decoder2(x, s2)
        x = self.decoder1(x, s1)
        
        # Output
        x = self.final_conv(x)
        return torch.sigmoid(x)

    def forward_with_features(self, x):
        """Return both segmentation and bottleneck features for classification"""
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        
        bottleneck = self.bottleneck(x)
        
        x = self.decoder4(bottleneck, s4)
        x = self.decoder3(x, s3)
        x = self.decoder2(x, s2)
        x = self.decoder1(x, s1)
        
        x = self.final_conv(x)
        return torch.sigmoid(x), bottleneck


# Alias for compatibility
AttentionUNet = AttentionUNetLegacy


def create_model(in_channels=1, out_channels=1):
    return AttentionUNetLegacy(in_channels=in_channels, out_channels=out_channels)


if __name__ == "__main__":
    model = AttentionUNetLegacy()
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
