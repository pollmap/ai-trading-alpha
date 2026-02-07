#!/usr/bin/env python3
"""Initialize the ATLAS database schema."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.data.db import init_schema, close_engine

log = get_logger(__name__)


async def main() -> None:
    setup_logging()
    log.info("starting_schema_initialization")

    try:
        await init_schema()
        log.info("schema_initialization_complete")
    except Exception as exc:
        log.error("schema_initialization_failed", error=str(exc))
        raise
    finally:
        await close_engine()


if __name__ == "__main__":
    asyncio.run(main())
