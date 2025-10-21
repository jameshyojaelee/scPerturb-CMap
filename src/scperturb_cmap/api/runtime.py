from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from fastapi import HTTPException, status

from scperturb_cmap.api.settings import ApiSettings
from scperturb_cmap.data.lincs_loader import load_lincs_long

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_lincs_cached(path_str: str) -> Tuple[pd.DataFrame, float]:
    """
    Load the LINCS library and cache the in-memory DataFrame with timestamp.

    Args:
        path_str: Absolute string path to file or directory.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"LINCS library not found at {path}")

    if path.is_dir():
        try:
            import pyarrow.dataset as ds  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError("pyarrow is required to read partitioned LINCS datasets") from exc
        dataset = ds.dataset(str(path), format="parquet")
        table = dataset.to_table()
        frame = table.to_pandas()
    else:
        frame = load_lincs_long(str(path))

    return frame, time.time()


def reset_lincs_cache() -> None:
    """Clear the cached LINCS DataFrame (primarily used in tests)."""
    load_lincs_cached.cache_clear()


def get_lincs_library(config: ApiSettings, *, force_refresh: bool = False) -> pd.DataFrame:
    """Return the cached LINCS DataFrame, enforcing TTL and surfacing HTTP errors."""
    if force_refresh:
        reset_lincs_cache()

    path = config.lincs_path
    try:
        library_df, loaded_at = load_lincs_cached(str(path.resolve()))
    except FileNotFoundError as exc:
        logger.error("LINCS library missing: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to load LINCS library from %s", path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load LINCS library: {exc}",
        ) from exc

    ttl = config.cache_ttl_seconds
    if ttl > 0 and (time.time() - loaded_at) > ttl:
        logger.info("LINCS cache expired (TTL=%s seconds); refreshing.", ttl)
        reset_lincs_cache()
        library_df, _ = load_lincs_cached(str(path.resolve()))

    return library_df


def get_model_path(config: ApiSettings, required: bool) -> Optional[str]:
    """Return the model path if present; error if required but absent."""
    model_path = config.model_path
    if not required:
        return str(model_path) if model_path.exists() else None
    if not model_path.exists():
        msg = f"Metric model not found at {model_path}"
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    return str(model_path)


async def check_redis_connection(url: str) -> None:
    """Verify Redis connectivity for readiness checks."""
    try:
        import redis.asyncio as aioredis  # type: ignore
    except ImportError as exc:
        raise RuntimeError("redis package not installed for readiness checks") from exc

    client = aioredis.from_url(url)
    try:
        await client.ping()
    finally:
        await client.close()


async def check_postgres_connection(dsn: str) -> None:
    """Verify PostgreSQL connectivity for readiness checks."""
    try:
        import asyncpg  # type: ignore
    except ImportError:
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Neither asyncpg nor psycopg available for PostgreSQL readiness checks"
            ) from exc

        def _sync_probe() -> None:
            with psycopg.connect(dsn) as conn:  # type: ignore[attr-defined]
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")

        await asyncio.to_thread(_sync_probe)
        return

    conn = await asyncpg.connect(dsn)
    try:
        await conn.fetchval("SELECT 1")
    finally:
        await conn.close()
