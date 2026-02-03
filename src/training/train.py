import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.segmentation_dataset import DentalSegmentationDataset
from src.models.attention_unet import AttentionUNet
from src.training.losses import CombinedLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DentalSegmentationDataset(
        image_dir="data/processed/train/images",
        mask_dir="data/splits/train/masks"
    )

    val_dataset = DentalSegmentationDataset(
        image_dir="data/processed/val/images",
        mask_dir="data/splits/val/masks"
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("results/checkpoints", exist_ok=True)

    epochs = 2
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                "results/checkpoints/best_model.pth"
            )
            print("Checkpoint saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
