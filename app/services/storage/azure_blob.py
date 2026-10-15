from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob.aio import BlobServiceClient

from app.core.config import Settings
from app.services.storage.base import StorageService


class AzureBlobStorageService(StorageService):
    def __init__(self, settings: Settings):
        self._container_name = settings.azure_container_name
        self._service_client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )

    async def download_image(self, key: str) -> bytes:
        container = self._service_client.get_container_client(self._container_name)
        blob = container.get_blob_client(key)
        try:
            stream = await blob.download_blob()
            return await stream.readall()
        except ResourceNotFoundError as exc:
            raise FileNotFoundError(f"Azure blob not found: {key}") from exc

    async def upload_image(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        container = self._service_client.get_container_client(self._container_name)
        blob = container.get_blob_client(key)
        await blob.upload_blob(data, overwrite=True, content_type=content_type)
        return blob.url
