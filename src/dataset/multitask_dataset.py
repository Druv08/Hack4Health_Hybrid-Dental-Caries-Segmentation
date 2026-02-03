# dataset/multitask_dataset.py
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import random
import torchvision.transforms as T

class MultiTaskDataset(Dataset):
    """
    Multitask dataset for dental caries:
    - Segmentation mask
    - Classification label (0 = normal, 1 = caries)
    
    Returns:
        img: [1, H, W] grayscale image
        mask: [1, H, W] segmentation mask
        label: 0 or 1
    """

    def __init__(self, split="train", img_dir="dataset/images", mask_dir="dataset/masks", augment=True):
        self.split = split
        self.img_dir = os.path.join(img_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)
        self.augment = augment

        self.images = sorted(os.listdir(self.img_dir))
        self.masks  = sorted(os.listdir(self.mask_dir))

        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

        # Define simple torchvision transforms for augmentation
        self.train_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load grayscale image & mask
        img_path  = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize to [0,1]
        img  = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Add channel dimension
        img  = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Classification label: 1 if any caries pixels
        label = 1 if mask.sum() > 0 else 0

        # Convert to tensors first
        img_tensor  = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply augmentation only for training
        if self.split == "train" and self.augment:
            # Combine image + mask for joint transforms
            combined = torch.cat([img_tensor, mask_tensor], dim=0)  # [2, H, W]
            combined = self.train_transforms(combined)
            img_tensor  = combined[0].unsqueeze(0)
            mask_tensor = combined[1].unsqueeze(0)

        return img_tensor, mask_tensor, label_tensor

