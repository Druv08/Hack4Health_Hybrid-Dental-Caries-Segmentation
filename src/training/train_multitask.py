import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ===== IMPORT YOUR EXISTING STUFF =====
from models.unet_attention import AttentionUNet
from models.classification_head import ClassificationHead

from metrics.segmentation_metrics import (
    dice_score,
    iou_score,
    pixel_accuracy,
    sensitivity,
    specificity,
    hausdorff_distance
)

from metrics.classification_metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    auc_score,
    confusion_matrix_score
)

from dataset.multitask_dataset import MultiTaskDataset

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== FAST + SAFE HYPERPARAMETERS =====
EPOCHS = 10          # keeps training under ~20 mins
BATCH_SIZE = 8
LR = 1e-4

# ===== DATA =====
train_dataset = MultiTaskDataset(split="train")
val_dataset   = MultiTaskDataset(split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== MODELS =====
seg_model = AttentionUNet().to(device)
clf_head  = ClassificationHead(in_channels=1024, num_classes=2).to(device)

# ===== LOSSES =====
seg_loss_fn = nn.BCELoss()
clf_loss_fn = nn.CrossEntropyLoss()

# ===== OPTIMIZER =====
optimizer = optim.Adam(
    list(seg_model.parameters()) + list(clf_head.parameters()),
    lr=LR
)

# ===== TRAINING =====
for epoch in range(EPOCHS):
    seg_model.train()
    clf_head.train()

    train_loss = 0.0

    for imgs, masks, labels in train_loader:
        imgs   = imgs.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ---- SEGMENTATION ----
        seg_preds, bottleneck = seg_model.forward_with_features(imgs)
        seg_loss = seg_loss_fn(seg_preds, masks)

        # ---- CLASSIFICATION ----
        cls_logits = clf_head(bottleneck)
        clf_loss = clf_loss_fn(cls_logits, labels)

        # ---- TOTAL LOSS (seg dominant) ----
        loss = seg_loss + 0.5 * clf_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ===== VALIDATION =====
    seg_model.eval()
    clf_head.eval()

    dice = iou = pix_acc = sens = spec = hd = 0
    acc = prec = rec = f1 = auc = 0
    cm = None

    with torch.no_grad():
        for imgs, masks, labels in val_loader:
            imgs   = imgs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            seg_preds, bottleneck = seg_model.forward_with_features(imgs)
            cls_logits = clf_head(bottleneck)

            # ---- SEG METRICS ----
            dice += dice_score(seg_preds, masks)
            iou  += iou_score(seg_preds, masks)
            pix_acc += pixel_accuracy(seg_preds, masks)
            sens += sensitivity(seg_preds, masks)
            spec += specificity(seg_preds, masks)
            hd   += hausdorff_distance(seg_preds, masks)

            # ---- CLS METRICS ----
            acc  += accuracy_score(cls_logits, labels)
            prec += precision_score(cls_logits, labels)
            rec  += recall_score(cls_logits, labels)
            f1   += f1_score(cls_logits, labels)
            auc  += auc_score(cls_logits, labels)
            cm   = confusion_matrix_score(cls_logits, labels)

    n = len(val_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")

    print("Segmentation Metrics:")
    print(f" Dice: {dice/n:.4f} | IoU: {iou/n:.4f} | Pixel Acc: {pix_acc/n:.4f}")
    print(f" Sensitivity: {sens/n:.4f} | Specificity: {spec/n:.4f}")
    print(f" Hausdorff Distance: {hd/n:.4f}")

    print("Classification Metrics:")
    print(f" Accuracy: {acc/n:.4f} | Precision: {prec/n:.4f}")
    print(f" Recall: {rec/n:.4f} | F1: {f1/n:.4f} | AUC: {auc/n:.4f}")
    print(f" Confusion Matrix:\n{cm}")

