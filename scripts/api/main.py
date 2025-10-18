"""
FastAPI-based REST API for scPerturb-CMap.

Exposes connectivity scoring endpoints with cached LINCS library loading
and basic observability hooks for production deployments.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.api.settings import ApiSettings, get_api_settings
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

APP_VERSION = "0.2.0"
settings = get_api_settings()
CONTENT_TOO_LARGE = (
    status.HTTP_413_CONTENT_TOO_LARGE
    if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE")
    else status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
)


def _runtime_settings() -> ApiSettings:
    """Return the cached API settings (used before app.state is available)."""
    return settings


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed the configured payload size."""

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_size:
                    detail = f"Request payload exceeds {self.max_body_size} bytes"
                    status_code = CONTENT_TOO_LARGE
                    return JSONResponse(
                        status_code=status_code,
                        content={
                            "error": {
                                "type": "http_error",
                                "message": detail,
                                "status": status_code,
                            }
                        },
                    )
            except ValueError:
                logger.debug(
                    "Unable to parse content-length header '%s'; allowing request to proceed",
                    content_length,
                )

        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Abort requests that exceed the configured execution timeout."""

    def __init__(self, app: ASGIApp, timeout: float) -> None:
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request exceeded the configured timeout limit",
            ) from exc


# Pydantic models for API
class ScoringRequest(BaseModel):
    """Request model for scoring endpoint."""

    target: dict = Field(..., description="Target signature with genes and weights")
    method: str = Field("baseline", description="Scoring method: baseline or metric")
    top_k: int = Field(50, ge=1, le=1000, description="Number of top results to return")
    cell_line: Optional[str] = Field(None, description="Filter by cell line")
    blend: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor for metric method")
    auto_blend: bool = Field(False, description="Automatically optimize blend factor")


class ScoringResponse(BaseModel):
    """Response model for scoring endpoint."""

    method: str
    ranking: list
    metadata: dict
    execution_time: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime: float


@lru_cache(maxsize=1)
def _load_lincs_cached(path_str: str) -> Tuple[pd.DataFrame, float]:
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
            raise RuntimeError(
                "pyarrow is required to read partitioned LINCS datasets"
            ) from exc
        dataset = ds.dataset(str(path), format="parquet")
        table = dataset.to_table()
        frame = table.to_pandas()
    else:
        frame = load_lincs_long(str(path))

    return frame, time.time()


def get_lincs_library(config: ApiSettings, *, force_refresh: bool = False) -> pd.DataFrame:
    """Return the cached LINCS DataFrame, enforcing TTL and surfacing HTTP errors."""
    if force_refresh:
        _load_lincs_cached.cache_clear()

    path = config.lincs_path
    try:
        library_df, loaded_at = _load_lincs_cached(str(path.resolve()))
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
        _load_lincs_cached.cache_clear()
        library_df, _ = _load_lincs_cached(str(path.resolve()))

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


async def _check_redis_connection(url: str) -> None:
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


async def _check_postgres_connection(dsn: str) -> None:
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


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info("Starting scPerturb-CMap API server...")

    runtime_settings = _runtime_settings()
    app.state.settings = runtime_settings
    app.state.start_time = time.time()
    app.state.metrics = get_metrics_collector(
        backend=runtime_settings.metrics_backend,
        port=runtime_settings.metrics_port,
        namespace=runtime_settings.metrics_namespace,
    )

    # Warm the LINCS cache to catch issues early (best effort)
    try:
        get_lincs_library(runtime_settings)
    except HTTPException as exc:
        # Defer error handling to request time; log for visibility.
        logger.warning("LINCS library warmup failed; will retry on demand. Detail: %s", exc.detail)

    try:
        yield
    finally:
        logger.info("Shutting down scPerturb-CMap API server...")


def _cors_origins_for(config: ApiSettings) -> List[str]:
    """Determine CORS origins based on environment."""
    if config.is_development and not config.cors_origins:
        return ["*"]
    if config.is_development and "*" in config.cors_origins:
        return ["*"]
    return config.cors_origins


# Create FastAPI app
app = FastAPI(
    title="scPerturb-CMap API",
    description="REST API for single-cell connectivity mapping",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Middleware configuration
cors_origins = _cors_origins_for(settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTimeoutMiddleware, timeout=settings.request_timeout_seconds)
app.add_middleware(BodySizeLimitMiddleware, max_body_size=settings.max_request_bytes)


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    metrics = getattr(app.state, "metrics", None)

    if metrics and hasattr(metrics, "active_requests"):
        metrics.active_requests.inc()

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    response = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    finally:
        duration = time.time() - start_time
        if metrics and hasattr(metrics, "active_requests"):
            metrics.active_requests.dec()
        if metrics:
            metrics.record_http_request(
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration=duration,
            )
        if response is not None:
            response.headers["X-Process-Time"] = f"{duration:.6f}"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return consistent validation error payloads."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return consistent HTTP error payloads."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status": exc.status_code,
            }
        },
    )


@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
    """Return a clear timeout response."""
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": {
                "type": "timeout",
                "message": "Request exceeded the configured timeout limit",
            }
        },
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        uptime=uptime,
    )


# Readiness probe
@app.get("/ready", tags=["System"])
@app.get("/api/ready", tags=["System"])
async def readiness_check():
    """Readiness probe for orchestrators."""
    runtime_settings = getattr(app.state, "settings", settings)

    checks: Dict[str, Dict[str, Optional[str]]] = {}
    errors: List[str] = []

    # LINCS dataset availability
    try:
        get_lincs_library(runtime_settings)
        checks["lincs"] = {"status": "ok"}
    except HTTPException as exc:
        checks["lincs"] = {"status": "error", "detail": exc.detail}
        errors.append(f"lincs: {exc.detail}")

    # Model availability (optional unless required)
    try:
        model_path = get_model_path(runtime_settings, required=runtime_settings.readiness_require_model)
        if model_path:
            checks["model"] = {"status": "ok", "path": model_path}
        else:
            checks["model"] = {"status": "skipped", "detail": "No model configured"}
    except HTTPException as exc:
        checks["model"] = {"status": "error", "detail": exc.detail}
        errors.append(f"model: {exc.detail}")

    # Redis connectivity (optional)
    if runtime_settings.redis_url and runtime_settings.readiness_check_redis:
        try:
            await _check_redis_connection(runtime_settings.redis_url)
            checks["redis"] = {"status": "ok"}
        except Exception as exc:  # pragma: no cover - external dependency
            detail = str(exc)
            checks["redis"] = {"status": "error", "detail": detail}
            errors.append(f"redis: {detail}")
    else:
        checks["redis"] = {"status": "skipped", "detail": "Not configured"}

    # PostgreSQL connectivity (optional)
    if runtime_settings.postgres_dsn and runtime_settings.readiness_check_postgres:
        try:
            await _check_postgres_connection(runtime_settings.postgres_dsn)
            checks["postgres"] = {"status": "ok"}
        except Exception as exc:  # pragma: no cover - external dependency
            detail = str(exc)
            checks["postgres"] = {"status": "error", "detail": detail}
            errors.append(f"postgres: {detail}")
    else:
        checks["postgres"] = {"status": "skipped", "detail": "Not configured"}

    if errors:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unready", "checks": checks, "errors": errors},
        )

    return {"status": "ready", "checks": checks}


# Metrics endpoint
@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint placeholder."""
    runtime_settings = getattr(app.state, "settings", settings)
    return {
        "message": "Metrics served via prometheus-client HTTP server",
        "backend": runtime_settings.metrics_backend,
        "port": runtime_settings.metrics_port,
    }


# Scoring endpoint
@app.post("/api/score", response_model=ScoringResponse, tags=["Scoring"])
async def score_target(
    request: ScoringRequest,
    background_tasks: BackgroundTasks,
):
    """
    Score a target signature against the LINCS library.

    Returns a ranked list of compounds with connectivity scores.
    """
    runtime_settings = getattr(app.state, "settings", settings)
    start_time = time.time()
    metrics = getattr(app.state, "metrics", None)

    try:
        # Parse target signature
        target_info = request.target
        if not isinstance(target_info, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`target` must be an object with genes and weights.",
            )

        try:
            target = TargetSignature(
                genes=target_info["genes"],
                weights=target_info["weights"],
                metadata=target_info.get("metadata", {}),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing target field: {exc}",
            ) from exc

        library_df = get_lincs_library(runtime_settings)
        model_path = get_model_path(
            runtime_settings,
            required=request.method.lower() == "metric",
        )

        # Perform scoring
        result = rank_drugs(
            target_signature=target,
            library=library_df,
            method=request.method,
            model_path=model_path,
            top_k=request.top_k,
            blend=request.blend,
            auto_blend=request.auto_blend,
        )

        # Convert to response format
        ranking_json = result.ranking.to_dict(orient="records")
        execution_time = time.time() - start_time

        if metrics:
            background_tasks.add_task(
                metrics.record_scoring_operation,
                method=request.method,
                cell_line=request.cell_line,
                duration=execution_time,
                success=True,
            )

        return ScoringResponse(
            method=result.method,
            ranking=ranking_json,
            metadata={**result.metadata, "cell_line": request.cell_line},
            execution_time=execution_time,
        )

    except HTTPException:
        if metrics:
            background_tasks.add_task(
                metrics.record_scoring_operation,
                method=request.method,
                cell_line=request.cell_line,
                duration=time.time() - start_time,
                success=False,
            )
        raise
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Scoring failed with an unexpected error.")
        if metrics:
            background_tasks.add_task(
                metrics.record_scoring_operation,
                method=request.method,
                cell_line=request.cell_line,
                duration=time.time() - start_time,
                success=False,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {
        "service": "scPerturb-CMap API",
        "version": APP_VERSION,
        "docs": "/api/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
    )
