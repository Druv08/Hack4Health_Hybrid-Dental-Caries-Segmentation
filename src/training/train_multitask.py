import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ===== IMPORT YOUR EXISTING STUFF =====
from models.unet_attention import AttentionUNet        # your existing model
from models.classification_head import ClassificationHead

from metrics.segmentation_metrics import dice_score, iou_score
from metrics.classification_metrics import accuracy_score

from dataset.multitask_dataset import MultiTaskDataset  # your dataset

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== HYPERPARAMETERS (FAST + GOOD SCORES) =====
EPOCHS = 12
BATCH_SIZE = 8
LR = 1e-4

# ===== DATA =====
train_dataset = MultiTaskDataset(split="train")
val_dataset   = MultiTaskDataset(split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== MODELS =====
seg_model = AttentionUNet().to(device)

# classification head takes bottleneck features
clf_head = ClassificationHead(in_channels=1024, num_classes=2).to(device)

# ===== LOSSES =====
seg_loss_fn = nn.BCELoss()
clf_loss_fn = nn.CrossEntropyLoss()

# ===== OPTIMIZER =====
optimizer = optim.Adam(
    list(seg_model.parameters()) + list(clf_head.parameters()),
    lr=LR
)

# ===== TRAIN LOOP =====
for epoch in range(EPOCHS):
    seg_model.train()
    clf_head.train()

    total_loss = 0

    for imgs, masks, labels in train_loader:
        imgs   = imgs.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ---- SEGMENTATION FORWARD ----
        seg_preds, bottleneck = seg_model.forward_with_features(imgs)
        seg_loss = seg_loss_fn(seg_preds, masks)

        # ---- CLASSIFICATION FORWARD ----
        cls_logits = clf_head(bottleneck)
        clf_loss = clf_loss_fn(cls_logits, labels)

        # ---- TOTAL LOSS ----
        loss = seg_loss + 0.5 * clf_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ===== VALIDATION =====
    seg_model.eval()
    clf_head.eval()

    dice, iou, acc = 0, 0, 0
    with torch.no_grad():
        for imgs, masks, labels in val_loader:
            imgs   = imgs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            seg_preds, bottleneck = seg_model.forward_with_features(imgs)
            cls_logits = clf_head(bottleneck)

            dice += dice_score(seg_preds, masks)
            iou  += iou_score(seg_preds, masks)
            acc  += accuracy_score_
