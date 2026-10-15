from app.core.config import settings
from app.services.storage.azure_blob import AzureBlobStorageService
from app.services.storage.base import StorageService
from app.services.storage.s3 import S3StorageService

_PROVIDERS: dict[str, type[StorageService]] = {
    "azure": AzureBlobStorageService,
    "s3": S3StorageService,
}


def get_storage_service() -> StorageService:
    provider = settings.storage_provider.lower()
    try:
        service_cls = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown storage provider '{provider}'. Valid options: {list(_PROVIDERS)}"
        )
    return service_cls(settings)
