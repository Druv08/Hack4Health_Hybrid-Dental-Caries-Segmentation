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
    def __init__(self, gate_channels, feature_channels, intermediate_channels):
        super(AttentionGate, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(feature_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g = self.W_gate(gate)
        x = self.W_x(skip_connection)
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        combined = self.relu(g + x)
        attention = self.psi(combined)
        return skip_connection * attention


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            gate_channels=in_channels // 2,
            feature_channels=skip_channels,
            intermediate_channels=skip_channels // 2
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        skip_attended = self.attention(x, skip)
        x = torch.cat([x, skip_attended], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        self.decoder4 = DecoderBlock(features[3] * 2, features[3], features[3])
        self.decoder3 = DecoderBlock(features[3], features[2], features[2])
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ===== Standard segmentation forward =====
    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x

    # ===== Multi-task forward (returns bottleneck for classification head) =====
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(in_channels=1, out_channels=1, pretrained=False):
    model = AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")
    return model


# ===== TESTING =====
if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, out_channels=1)
    print(f"Model created successfully. Total params: {model.get_num_parameters():,}")
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape == (4, 1, 256, 256)
    assert output.min() >= 0 and output.max() <= 1
    print("[OK] All tests passed!")

