# models/unet_attention.py
"""
Attention U-Net Model for Dental Caries Segmentation
=====================================================

Improvements:
- Increased channels: 64 → 128 → 256 → 512 → 1024
- Residual connections in ConvBlocks
- Dropout in bottleneck
- Multi-task forward for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two conv layers + BatchNorm + ReLU with residual connection"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        # Residual conv if in/out channels mismatch
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.res_conv(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + res)  # residual connection
        return x


class AttentionGate(nn.Module):
    """Attention gate for skip connection"""
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
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g + x)
        attention = self.psi(psi)
        return skip * attention


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        pooled = self.pool(feat)
        return feat, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(in_channels // 2, skip_channels, skip_channels // 2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip_att = self.attention(x, skip)
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024], dropout=0.3):
        super(AttentionUNet, self).__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, features[0])
        self.enc2 = EncoderBlock(features[0], features[1])
        self.enc3 = EncoderBlock(features[1], features[2])
        self.enc4 = EncoderBlock(features[2], features[3])
        self.enc5 = EncoderBlock(features[3], features[4])

        # Bottleneck with dropout
        self.bottleneck = nn.Sequential(
            ConvBlock(features[4], features[4]),
            nn.Dropout2d(dropout)
        )

        # Decoder
        self.dec5 = DecoderBlock(features[4], features[4], features[3])
        self.dec4 = DecoderBlock(features[3], features[3], features[2])
        self.dec3 = DecoderBlock(features[2], features[2], features[1])
        self.dec2 = DecoderBlock(features[1], features[1], features[0])
        self.dec1 = DecoderBlock(features[0], features[0], features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # Standard segmentation forward
    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        s5, x = self.enc5(x)

        x = self.bottleneck(x)

        x = self.dec5(x, s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.final_conv(x)
        return torch.sigmoid(x)

    # Multi-task forward (return bottleneck for classification)
    def forward_with_features(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        s5, x = self.enc5(x)

        bottleneck = self.bottleneck(x)

        x = self.dec5(bottleneck, s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.final_conv(x)
        return torch.sigmoid(x), bottleneck

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(in_channels=1, out_channels=1, pretrained=False):
    model = AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available")
    return model


# ===== Test =====
if __name__ == "__main__":
    model = AttentionUNet()
    print(f"Model params: {model.get_num_parameters():,}")
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        out, bottleneck = model.forward_with_features(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}, Bottleneck shape: {bottleneck.shape}")
    assert out.shape == (4, 1, 256, 256)
    print("[OK] AttentionUNet ready for multitask training")
