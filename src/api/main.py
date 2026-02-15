"""ATLAS FastAPI application — entry point for the API server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from src.core.logging import get_logger
from src.data.db import close_engine, get_engine

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle — initialize DB engine, close on exit."""
    log.info("api_starting")
    await get_engine()
    yield
    await close_engine()
    log.info("api_shutdown")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ATLAS API",
        description="AI Trading Lab for Agent Strategy — REST API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url, "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from src.api.routes.auth import router as auth_router
    from src.api.routes.health import router as health_router
    from src.api.routes.portfolios import router as portfolios_router
    from src.api.routes.simulations import router as simulations_router
    from src.api.routes.strategies import router as strategies_router
    from src.api.routes.usage import router as usage_router

    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(portfolios_router, prefix="/api")
    app.include_router(simulations_router, prefix="/api")
    app.include_router(strategies_router, prefix="/api")
    app.include_router(usage_router, prefix="/api")

    return app


app = create_app()
