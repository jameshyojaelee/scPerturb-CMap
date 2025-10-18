from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "on"}:
        return True
    if value_normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Unrecognised boolean value '%s'; using default=%s", value, default)
    return default


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Unable to parse integer value '%s'; using default=%s", value, default)
        return default


def _parse_bytes_limit(
    bytes_value: Optional[str],
    mb_value: Optional[str],
    default_bytes: int,
) -> int:
    if bytes_value is not None:
        parsed = _parse_int(bytes_value, default_bytes)
        return parsed if parsed > 0 else default_bytes
    if mb_value is not None:
        parsed = _parse_int(mb_value, default_bytes // (1024 * 1024))
        computed = parsed * 1024 * 1024
        return computed if computed > 0 else default_bytes
    return default_bytes


def _parse_origins(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


class ApiSettings(BaseModel):
    """Configuration for the public FastAPI service."""

    environment: str = Field(default="production")
    lincs_path: Path = Field(default=Path("/data/lincs/partitioned"))
    model_path: Path = Field(default=Path("/app/workspace/artifacts/best.pt"))
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    metrics_backend: str = Field(default="prometheus")
    metrics_port: int = Field(default=8000, ge=0)
    metrics_namespace: Optional[str] = None
    redis_url: Optional[str] = None
    postgres_dsn: Optional[str] = None
    request_timeout_seconds: int = Field(default=30, ge=1)
    max_request_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    readiness_require_model: bool = Field(default=False)
    readiness_check_redis: bool = Field(default=True)
    readiness_check_postgres: bool = Field(default=True)

    @property
    def is_development(self) -> bool:
        return self.environment.lower() in {"dev", "development", "local"}

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        env = os.environ

        environment = env.get("SCPC_ENV") or env.get("ENVIRONMENT") or "production"
        lincs_path = env.get("SCPC_LINCS_PATH") or str(Path("/data/lincs/partitioned"))
        model_path = env.get("SCPC_MODEL_PATH") or str(Path("/app/workspace/artifacts/best.pt"))
        cache_ttl = _parse_int(env.get("SCPC_CACHE_TTL"), 3600)

        cors_origins = _parse_origins(env.get("SCPC_CORS_ORIGINS"))
        if not cors_origins:
            cors_origins = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]

        metrics_backend = (
            env.get("SCPC_METRICS_BACKEND")
            or env.get("METRICS_BACKEND")
            or "prometheus"
        )
        metrics_port = _parse_int(
            env.get("SCPC_METRICS_PORT") or env.get("METRICS_PORT"),
            8000,
        )
        metrics_namespace = env.get("SCPC_METRICS_NAMESPACE")

        redis_url = env.get("SCPC_REDIS_URL") or env.get("REDIS_URL")
        postgres_dsn = env.get("SCPC_DATABASE_URL") or env.get("DATABASE_URL")

        request_timeout_seconds = _parse_int(env.get("SCPC_REQUEST_TIMEOUT"), 30)
        max_request_bytes = _parse_bytes_limit(
            bytes_value=env.get("SCPC_MAX_REQUEST_BYTES"),
            mb_value=env.get("SCPC_MAX_REQUEST_SIZE_MB"),
            default_bytes=25 * 1024 * 1024,
        )

        readiness_require_model = _parse_bool(env.get("SCPC_REQUIRE_MODEL"), False)
        readiness_check_redis = _parse_bool(env.get("SCPC_READINESS_CHECK_REDIS"), True)
        readiness_check_postgres = _parse_bool(env.get("SCPC_READINESS_CHECK_POSTGRES"), True)

        try:
            return cls.model_validate(
                {
                    "environment": environment,
                    "lincs_path": Path(lincs_path).expanduser(),
                    "model_path": Path(model_path).expanduser(),
                    "cache_ttl_seconds": cache_ttl,
                    "cors_origins": cors_origins,
                    "metrics_backend": metrics_backend,
                    "metrics_port": metrics_port,
                    "metrics_namespace": metrics_namespace,
                    "redis_url": redis_url,
                    "postgres_dsn": postgres_dsn,
                    "request_timeout_seconds": request_timeout_seconds,
                    "max_request_bytes": max_request_bytes,
                    "readiness_require_model": readiness_require_model,
                    "readiness_check_redis": readiness_check_redis,
                    "readiness_check_postgres": readiness_check_postgres,
                }
            )
        except ValidationError as exc:  # pragma: no cover - defensive logging
            logger.error("Invalid API settings: %s", exc)
            raise


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    """Load API settings from environment variables (cached)."""
    return ApiSettings.from_environment()


def reset_api_settings_cache() -> None:
    """Clear the cached API settings (used in tests)."""
    get_api_settings.cache_clear()
