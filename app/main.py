"""
main.py
FastAPI entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.images import router as images_router
from app.core.config import settings
from app.services.vision.inference import get_predictor

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the singleton predictor once, instead of loading the
    # model on every request (as the notebook implicitly did by keeping
    # everything in one long-running process).
    get_predictor()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(images_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
