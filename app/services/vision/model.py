"""
services/vision/model.py
FusionDeepLab: DeepLabV3+ (ImageNet-pretrained encoder) with an added
color-texture fusion branch blended into the encoder's layer3 features.
Moved verbatim (logic-wise) from the notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from app.services.vision.fusion import compute_color_texture_features, FusionAdapter


class FusionDeepLab(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", num_classes: int = 3, alpha: float = 0.5, encoder_weights: str | None = "imagenet"):
        """
        alpha = blending weight between RGB encoder features and fused
        color-texture features.
        """
        super().__init__()
        self.alpha = alpha

        # Pretrained backbone
        self.base = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        # Fusion adapter (7 fused channels -> 64 -> projected to 256 to match layer3)
        self.adapter = FusionAdapter(in_ch=7, out_ch=64)
        self.proj = nn.Conv2d(64, 256, kernel_size=1)

        # Hook to capture encoder layer3 output
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(_, __, output):
            self.encoder_feat = output

        getattr(self.base.encoder, "layer3").register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base DeepLab pass
        out = self.base(x)

        # Color-texture fusion
        fused = compute_color_texture_features(x)
        fused_feat = self.adapter(fused)
        fused_feat = self.proj(fused_feat)

        # Blend fusion into captured encoder features (resize if necessary)
        if hasattr(self, "encoder_feat"):
            if fused_feat.shape[2:] != self.encoder_feat.shape[2:]:
                fused_feat = F.interpolate(
                    fused_feat, size=self.encoder_feat.shape[2:], mode="bilinear", align_corners=False
                )
            blended = (1 - self.alpha) * self.encoder_feat + self.alpha * fused_feat
            self.encoder_feat = blended  # replace in-place for decoder use

        return out
