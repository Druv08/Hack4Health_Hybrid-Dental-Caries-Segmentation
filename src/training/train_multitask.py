# train_multitask.py
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
    precision,
    recall,
    f1score,
    auc_score,
    conf_matrix
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

    # --- Initialize metrics ---
    dice_total = iou_total = 0
    pix_acc_total = sens_total = spec_total = hd_total = 0
    acc_total = prec_total = rec_total = f1_total = auc_total = 0
    all_conf_matrices = []

    with torch.no_grad():
        for imgs, masks, labels in val_loader:
            imgs   = imgs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            # Forward pass
            seg_preds, bottleneck = seg_model.forward_with_features(imgs)
            cls_logits = clf_head(bottleneck)

            # ---- SEGMENTATION METRICS ----
            dice_total += dice_score(seg_preds, masks)
            iou_total  += iou_score(seg_preds, masks)
            pix_acc_total += pixel_accuracy(seg_preds, masks)
            sens_total += sensitivity(seg_preds, masks)
            spec_total += specificity(seg_preds, masks)
            hd_total += hausdorff_distance(seg_preds, masks)

            # ---- CLASSIFICATION METRICS ----
            acc_total  += accuracy_score(cls_logits, labels)
            prec_total += precision(cls_logits, labels)
            rec_total  += recall(cls_logits, labels)
            f1_total   += f1score(cls_logits, labels)
            auc_total  += auc_score(cls_logits, labels)
            all_conf_matrices.append(conf_matrix(cls_logits, labels))

    # --- Average metrics over batches ---
    num_batches = len(val_loader)

    print(f"\n===== Epoch [{epoch+1}/{EPOCHS}] =====")
    print(f"Train Loss: {train_loss:.4f}")

    print("\nSegmentation Metrics:")
    print(f" Dice: {dice_total/num_batches:.4f} | IoU: {iou_total/num_batches:.4f} | Pixel Acc: {pix_acc_total/num_batches:.4f}")
    print(f" Sensitivity: {sens_total/num_batches:.4f} | Specificity: {spec_total/num_batches:.4f}")
    print(f" Hausdorff Distance: {hd_total/num_batches:.4f}")

    print("\nClassification Metrics:")
    print(f" Accuracy: {acc_total/num_batches:.4f} | Precision: {prec_total/num_batches:.4f}")
    print(f" Recall: {rec_total/num_batches:.4f} | F1: {f1_total/num_batches:.4f} | AUC: {auc_total/num_batches:.4f}")
    print(f"Confusion Matrices per batch: {all_conf_matrices}")


