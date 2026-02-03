import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class MultiTaskDataset(Dataset):
    """
    Dataset for multitask dental caries segmentation + classification.
    Returns: image, mask, label
    """

    def __init__(self, split="train", img_dir="dataset/images", mask_dir="dataset/masks", transform=None):
        """
        Args:
            split (str): 'train' or 'val'
            img_dir (str): Directory containing X-ray images
            mask_dir (str): Directory containing segmentation masks
            transform (callable, optional): Optional transforms on image/mask
        """
        self.split = split
        self.img_dir = os.path.join(img_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)
        self.transform = transform

        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize image to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Classification label: 1 if mask has any caries pixels
        label = 1 if mask.sum() > 0 else 0

        # Apply transforms if any
        if self.transform:
            img, mask = self.transform(img, mask)

        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)
