import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, segmentation_model, encoder_channels=512):
        super().__init__()
        
        # Existing segmentation model
        self.segmentation_model = segmentation_model
        
        # Classification head (VERY lightweight)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Segmentation output
        seg_out, encoder_features = self.segmentation_model(x, return_features=True)
        
        # Classification output
        cls_out = self.classifier(encoder_features)
        
        return seg_out, cls_out
