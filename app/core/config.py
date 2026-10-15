from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_name: str = "Burn Segmentation API"
    env: str = "development"

    # --- Storage backend selection: "s3" or "azure" ---
    storage_provider: str = "s3"

    # --- S3 ---
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str | None = None

    # --- Azure Blob ---
    azure_storage_connection_string: str | None = None
    azure_container_name: str | None = None

    # --- Model / inference ---
    model_checkpoint_path: str = "weights/fusion_deeplab.pt"
    model_encoder_name: str = "resnet34"
    model_num_classes: int = 3
    model_alpha: float = 0.5
    model_device: str = "cuda"  # falls back to "cpu" automatically if unavailable
    model_input_size: int = 768


settings = Settings()
