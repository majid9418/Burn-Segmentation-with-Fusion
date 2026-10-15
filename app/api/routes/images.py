"""
api/routes/images.py
HTTP layer: download image via storage backend -> run segmentation
inference -> return mask + per-class area stats.

This replaces the notebook's manual flow of:
    cv2.imread(local_path) -> transform -> model(input) -> plt.imshow(...)
with a storage-agnostic, request-driven pipeline.
"""
import base64

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.schemas.image import ClassArea, SegmentationRequest, SegmentationResponse
from app.services.storage.base import StorageService
from app.services.storage.factory import get_storage_service
from app.services.vision.inference import SegmentationPredictor, get_predictor

router = APIRouter(prefix="/images", tags=["images"])


def _encode_mask_png(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("Failed to encode predicted mask as PNG")
    return buf.tobytes()


def _class_areas(mask: np.ndarray, num_classes: int) -> list[ClassArea]:
    total = mask.size
    areas = []
    for cls in range(num_classes):
        count = int((mask == cls).sum())
        areas.append(
            ClassArea(
                class_index=cls,
                pixel_count=count,
                percentage=round(100.0 * count / total, 2) if total else 0.0,
            )
        )
    return areas


@router.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    payload: SegmentationRequest,
    storage: StorageService = Depends(get_storage_service),
    predictor: SegmentationPredictor = Depends(get_predictor),
):
    try:
        image_bytes = await storage.download_image(payload.image_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        mask = predictor.predict(image_bytes)
    except Exception as exc:  # model/decoding failure
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    height, width = mask.shape
    mask_png_bytes = _encode_mask_png(mask)
    mask_base64 = base64.b64encode(mask_png_bytes).decode("utf-8")

    mask_url = None
    if payload.upload_result_mask:
        result_key = f"results/{payload.image_key}_mask.png"
        mask_url = await storage.upload_image(result_key, mask_png_bytes, content_type="image/png")

    return SegmentationResponse(
        image_key=payload.image_key,
        width=width,
        height=height,
        mask_base64=mask_base64,
        mask_url=mask_url,
        class_areas=_class_areas(mask, num_classes=predictor.num_classes),
    )
