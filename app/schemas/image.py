from pydantic import BaseModel, Field


class SegmentationRequest(BaseModel):
    image_key: str = Field(
        ..., description="Key/path of the source image inside the configured storage backend"
    )
    upload_result_mask: bool = Field(
        default=False,
        description="If true, the predicted mask is also uploaded to storage and its URL returned",
    )


class ClassArea(BaseModel):
    class_index: int
    pixel_count: int
    percentage: float


class SegmentationResponse(BaseModel):
    image_key: str
    width: int
    height: int
    mask_base64: str = Field(..., description="PNG-encoded predicted mask, base64-encoded")
    mask_url: str | None = Field(
        default=None, description="Storage URL of the uploaded mask, if upload_result_mask=true"
    )
    class_areas: list[ClassArea]
