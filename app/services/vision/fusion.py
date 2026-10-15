"""
services/vision/fusion.py
Color-texture fusion feature extraction (RGB + Lab + texture) and the
small adapter network that projects those 7 channels down for blending
into the encoder. Moved verbatim (logic-wise) from the notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as K


def compute_color_texture_features(images: torch.Tensor) -> torch.Tensor:
    """
    Input: ImageNet-normalized RGB tensor [B,3,H,W]
    Output: fused tensor [B,7,H,W] = RGB + LAB + texture
    """
    # --- De-normalize before LAB conversion ---
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    x_unnorm = (images * std) + mean
    x_unnorm = x_unnorm.clamp(0, 1)

    # --- Convert to LAB ---
    lab = K.rgb_to_lab(x_unnorm)
    lab = (lab - lab.mean(dim=(2, 3), keepdim=True)) / (lab.std(dim=(2, 3), keepdim=True) + 1e-6)

    # --- Compute texture variance ---
    gray = K.rgb_to_grayscale(x_unnorm)
    kernel = torch.ones((1, 1, 3, 3), device=images.device) / 9.0
    mean_local = F.conv2d(gray, kernel, padding=1)
    var_local = F.conv2d((gray - mean_local) ** 2, kernel, padding=1)
    texture = torch.sqrt(var_local + 1e-6)
    texture = (texture - texture.mean(dim=(2, 3), keepdim=True)) / (texture.std(dim=(2, 3), keepdim=True) + 1e-6)

    # --- Fuse channels: RGB + LAB + texture ---
    fused = torch.cat([x_unnorm, lab, texture], dim=1)  # [B,7,H,W]
    return fused


class FusionAdapter(nn.Module):
    """Light adapter with BatchNorm that projects the 7-channel fused
    tensor down to a channel count that matches the encoder's mid-level
    features."""

    def __init__(self, in_ch: int = 7, out_ch: int = 3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(x)
