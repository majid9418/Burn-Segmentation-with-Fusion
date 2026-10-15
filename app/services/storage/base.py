from abc import ABC, abstractmethod


class StorageService(ABC):
    """Abstraction over 'where the image bytes come from'."""

    @abstractmethod
    async def download_image(self, key: str) -> bytes:
        """Download raw image bytes given a key/path/identifier."""
        raise NotImplementedError

    @abstractmethod
    async def upload_image(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        """Upload raw bytes (e.g. a predicted mask) and return a retrievable URL/key."""
        raise NotImplementedError
