"""
services/vision/inference.py
Production inference path. The model is loaded ONCE (singleton) and
reused across requests — unlike the notebook, which built/trained the
model inline in the same script.
"""
from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import Settings, settings as default_settings
from app.services.vision.model import FusionDeepLab
from app.services.vision.transforms import get_transforms


class SegmentationPredictor:
    def __init__(self, settings: Settings):
        self.device = torch.device(
            settings.model_device if torch.cuda.is_available() else "cpu"
        )
        self.input_size = settings.model_input_size
        self.num_classes = settings.model_num_classes
        self.transform = get_transforms(input_size=self.input_size)

        self.model = FusionDeepLab(
            encoder_name=settings.model_encoder_name,
            num_classes=settings.model_num_classes,
            alpha=settings.model_alpha,
            encoder_weights=None,
        )
        checkpoint = torch.load(settings.model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def _bytes_to_rgb(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw bytes (downloaded from storage) into an RGB array,
        mirroring the notebook's cv2.imread + BGR2RGB step."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(image)

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> np.ndarray:
        """Run the full pipeline on one image and return a HxW class-index mask."""
        rgb = self._bytes_to_rgb(image_bytes)
        original_h, original_w = rgb.shape[:2]

        transformed = self.transform(image=rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        output = self.model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to the original image size
        # Undo LongestMaxSize + PadIfNeeded before resizing back to the original size
        scale = self.input_size / max(original_h, original_w)
        resized_h, resized_w = round(original_h * scale), round(original_w * scale)
        pad_top = (self.input_size - resized_h) // 2
        pad_left = (self.input_size - resized_w) // 2
        pred_mask = pred_mask[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]

        pred_mask = cv2.resize(
            pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )

        return pred_mask


@lru_cache
def get_predictor() -> SegmentationPredictor:
    """Singleton accessor — created once at first use (or app startup)."""
    return SegmentationPredictor(default_settings)
