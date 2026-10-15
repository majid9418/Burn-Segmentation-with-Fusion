"""
services/vision/dataset.py
Disk-based Dataset used only for offline training/evaluation
(scripts/train.py). Not used by the live API request path.
"""
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class BurnSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_list = sorted([
            fname for fname in os.listdir(image_dir)
            if fname.endswith(".png")
        ]) if image_list is None else image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        id_number = os.path.splitext(img_name)[0]
        label_name = f"Label_{id_number}.png"

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, label_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = np.clip(mask, 0, 2)

        # Resize mask to match image if needed
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long(), img_name
