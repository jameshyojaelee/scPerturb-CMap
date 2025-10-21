"""
FastAPI-based REST API for scPerturb-CMap.

Exposes connectivity scoring endpoints with cached LINCS library loading
and basic observability hooks for production deployments.
"""
from __future__ import annotations

import asyncio
import logging
import os
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from celery.exceptions import TimeoutError as CeleryTimeoutError
from celery.result import AsyncResult
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from scperturb_cmap.api.runtime import (
    check_postgres_connection,
    check_redis_connection,
    get_lincs_library,
    get_model_path,
)
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.api.settings import ApiSettings, get_api_settings
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.utils.metrics import get_metrics_collector
from scperturb_cmap.workers.tasks import score_target_task

logger = logging.getLogger(__name__)

APP_VERSION = "0.2.0"
settings = get_api_settings()
CONTENT_TOO_LARGE = (
    status.HTTP_413_CONTENT_TOO_LARGE
    if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE")
    else status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
)

CELERY_STATE_MAP = {
    "PENDING": "pending",
    "STARTED": "running",
    "RETRY": "retrying",
    "SUCCESS": "completed",
    "FAILURE": "failed",
    "REVOKED": "cancelled",
}


def _celery_state(state: Optional[str]) -> str:
    """Normalise Celery task states to API-friendly labels."""
    if not state:
        return "pending"
    return CELERY_STATE_MAP.get(state.upper(), state.lower())


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


def _get_celery_app_or_503():
    """Return the configured Celery app or raise a 503 if unavailable."""
    celery_app = getattr(app.state, "celery", None)
    queue_enabled = getattr(app.state, "queue_enabled", False)
    if not celery_app or not queue_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Asynchronous scoring workers are not configured.",
        )
    return celery_app


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


class JobSubmissionResponse(BaseModel):
    """Response returned after enqueueing a scoring job."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Status payload for a scoring job."""

    job_id: str
    status: str
    result: Optional[ScoringResponse] = None
    detail: Optional[str] = None


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

    celery_app = getattr(score_target_task, "app", None)
    if celery_app:
        app.state.celery = celery_app
    else:
        app.state.celery = None

    broker_present = runtime_settings.redis_url or os.getenv("CELERY_BROKER_URL")
    app.state.queue_enabled = bool(app.state.celery) and bool(broker_present)

    if app.state.queue_enabled:
        broker_url = getattr(app.state.celery.conf, "broker_url", "unknown")
        logger.info("Celery queue initialised (broker=%s).", broker_url)
    else:
        logger.info("Celery queue disabled; asynchronous scoring endpoints will return 503.")

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
            await check_redis_connection(runtime_settings.redis_url)
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
            await check_postgres_connection(runtime_settings.postgres_dsn)
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


# Asynchronous scoring endpoints
@app.post("/api/score/jobs", response_model=JobSubmissionResponse, tags=["Scoring"])
async def enqueue_score_job(request: ScoringRequest):
    """Enqueue a connectivity scoring job for background processing."""
    _get_celery_app_or_503()  # Ensure queue is configured
    payload = request.model_dump()
    async_result = score_target_task.apply_async(kwargs={"payload": payload})
    status_label = _celery_state(async_result.state)
    return JobSubmissionResponse(job_id=async_result.id, status=status_label)


@app.get("/api/score/jobs/{job_id}", response_model=JobStatusResponse, tags=["Scoring"])
async def get_score_job_status(
    job_id: str,
    wait: Optional[float] = Query(
        None,
        ge=0.0,
        le=30.0,
        description="Optional seconds to wait for job completion before returning.",
    ),
):
    """Return the status (and result when available) for a scoring job."""
    celery_app = _get_celery_app_or_503()
    async_result = AsyncResult(job_id, app=celery_app)

    if wait and wait > 0:
        try:
            await asyncio.to_thread(async_result.get, timeout=wait)
        except CeleryTimeoutError:
            pass

    state = _celery_state(async_result.state)

    if async_result.successful():
        result_payload = async_result.result
        response = JobStatusResponse(
            job_id=job_id,
            status=state,
            result=ScoringResponse(**result_payload),
        )
        return response

    if async_result.failed():
        detail = str(async_result.result)
        response = JobStatusResponse(job_id=job_id, status=state, detail=detail)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump(),
        )

    response = JobStatusResponse(job_id=job_id, status=state)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=response.model_dump(),
    )


@app.get("/api/score/jobs/{job_id}/stream", tags=["Scoring"])
async def stream_score_job(job_id: str, interval: float = Query(1.0, ge=0.1, le=5.0)):
    """Stream job status updates until the scoring result is available."""
    celery_app = _get_celery_app_or_503()

    async def event_stream():
        while True:
            async_result = AsyncResult(job_id, app=celery_app)
            state = _celery_state(async_result.state)
            message: Dict[str, Any] = {"job_id": job_id, "status": state}
            if async_result.successful():
                message["result"] = async_result.result
                yield json.dumps(message) + "\n"
                break
            if async_result.failed():
                message["detail"] = str(async_result.result)
                yield json.dumps(message) + "\n"
                break

            yield json.dumps(message) + "\n"
            await asyncio.sleep(interval)

    return StreamingResponse(event_stream(), media_type="application/jsonl")


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
