import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routes.images import router
from app.services.storage.base import StorageService
from app.services.vision.inference import get_predictor, SegmentationPredictor
from app.services.storage.factory import get_storage_service
from fastapi import FastAPI


class _FakeStorage(StorageService):
    async def download_image(self, key: str) -> bytes:
        # 4x4 white PNG, good enough for the fake predictor below
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format="PNG")
        return buf.getvalue()

    async def upload_image(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        return f"fake://{key}"


class _FakePredictor:
    num_classes = 3

    def predict(self, image_bytes: bytes) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.uint8)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage_service] = lambda: _FakeStorage()
    app.dependency_overrides[get_predictor] = lambda: _FakePredictor()
    return TestClient(app)


def test_segment_image_returns_200(client):
    response = client.post("/images/segment", json={"image_key": "sample.png"})
    assert response.status_code == 200
    body = response.json()
    assert body["image_key"] == "sample.png"
    assert body["width"] == 4
    assert body["height"] == 4
    assert len(body["class_areas"]) == 3


def test_segment_image_missing_key_returns_404(client):
    class _MissingStorage(_FakeStorage):
        async def download_image(self, key: str) -> bytes:
            raise FileNotFoundError(key)

    client.app.dependency_overrides[get_storage_service] = lambda: _MissingStorage()
    response = client.post("/images/segment", json={"image_key": "missing.png"})
    assert response.status_code == 404
