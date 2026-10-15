"""
services/vision/losses.py
Loss functions used for training. Not needed at inference time, but
kept alongside the model definition for cohesion with scripts/train.py.
"""
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)

        num_classes = outputs.shape[1]
        total_dice = 0

        for c in range(num_classes):
            pred_c = outputs[:, c, :, :]
            target_c = (targets == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_dice += 1 - dice

        return total_dice / num_classes
