"""
services/vision/transforms.py
Albumentations pipelines — moved verbatim (logic-wise) from the notebook.
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(input_size: int = 768) -> A.Compose:
    """Augmentation pipeline used during training."""
    return A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5,
        ),

        # Elastic deformation (preserves structure, good for skin textures)
        A.ElasticTransform(
            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,
            interpolation=1, border_mode=cv2.BORDER_CONSTANT,
            value=0, mask_value=0, p=0.3,
        ),

        # Brightness/contrast variation
        A.RandomBrightnessContrast(p=0.4),

        # Resize and pad
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),

        # Normalize + convert
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_transforms(input_size: int = 768) -> A.Compose:
    """Deterministic pipeline used for eval / inference (no augmentation)."""
    return A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      border_mode=0, value=0, mask_value=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
