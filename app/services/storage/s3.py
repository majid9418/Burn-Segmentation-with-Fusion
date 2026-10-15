import asyncio
from functools import partial

import boto3
from botocore.exceptions import ClientError

from app.core.config import Settings
from app.services.storage.base import StorageService


class S3StorageService(StorageService):
    def __init__(self, settings: Settings):
        self._bucket = settings.s3_bucket_name
        self._client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )

    async def download_image(self, key: str) -> bytes:
        loop = asyncio.get_running_loop()
        try:
            obj = await loop.run_in_executor(
                None, partial(self._client.get_object, Bucket=self._bucket, Key=key)
            )
            return obj["Body"].read()
        except ClientError as exc:
            raise FileNotFoundError(f"S3 object not found: {key}") from exc

    async def upload_image(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            ),
        )
        return f"s3://{self._bucket}/{key}"
