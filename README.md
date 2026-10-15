# Burn Segmentation API

FastAPI service that serves a burn-wound segmentation model (DeepLabV3+ with
an added color-texture fusion branch) behind a storage-agnostic HTTP layer.
Built on top of the team's `Python template` project skeleton.

## Structure

```
app/
  main.py                     FastAPI entry point (lifespan warms up the model)
  core/config.py              Env-based settings (singleton `settings`)
  api/routes/images.py        POST /images/segment
  schemas/image.py            Request/response models
  services/storage/           StorageService interface + S3 / Azure Blob impls
  services/vision/            Model, fusion branch, transforms, losses,
                               metrics, dataset — shared by the API and by
                               scripts/train.py
scripts/train.py              Offline training/evaluation entry point
tests/                        pytest suite (uses dependency_overrides to
                               fake storage + predictor)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in storage credentials + model paths
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Train (offline)

```bash
python -m scripts.train
```

Reads `BURN_IMAGE_DIR` / `BURN_MASK_DIR` / `MODEL_CHECKPOINT_PATH` from the
environment (see `scripts/train.py` for defaults) and writes a checkpoint
that `MODEL_CHECKPOINT_PATH` in `.env` should then point to.

## Notes on merging with the template

- `app/core/config.py` keeps the template's pattern: a single lowercase-field
  `Settings` class exported as a module-level `settings` instance (not a
  `get_settings()` factory), so `from app.core.config import settings` works
  the same way everywhere in this project as it does in the template.
- `StorageService` / `S3StorageService` / `AzureBlobStorageService` keep the
  template's class names. Only the abstract methods changed
  (`download_image`/`upload_image` instead of the template's placeholder
  `download_images`), because this service actually needs to fetch a single
  image and push back a predicted mask, not just log/mock URLs.
- `app/services/vision/*` and `scripts/train.py` are carried over unchanged
  from the original submission — they don't touch `app.core.config` directly
  in a way that needed renaming (only `inference.py` did, since it consumes
  `Settings`).
- Added `.env.example`, `requirements-dev.txt` (pinned versions), and a real
  `tests/test_images.py`, none of which existed in the original submission,
  to match what the template project provides.
