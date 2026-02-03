import torch
import torch.nn as nn

class MultiTaskWrapper(nn.Module):
    def __init__(self, segmentation_model, bottleneck_layer):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.features = None

        # Hook to capture bottleneck features
        def hook_fn(module, input, output):
            self.features = output

        bottleneck_layer.register_forward_hook(hook_fn)

        # Lightweight classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seg_out = self.segmentation_model(x)
        cls_out = self.classifier(self.features)
        return seg_out, cls_out
