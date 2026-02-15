"""Health check endpoint — no auth required."""

from __future__ import annotations

from fastapi import APIRouter

from config.settings import get_settings
from src.api.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.atlas_env,
    )
