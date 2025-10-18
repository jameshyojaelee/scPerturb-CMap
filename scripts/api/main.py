"""
FastAPI-based REST API for scPerturb-CMap.

Exposes connectivity scoring endpoints with cached LINCS library loading
and basic observability hooks for production deployments.
"""
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional
import logging
import os
import time

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

DEFAULT_LINCS_PATH = "/data/lincs/partitioned"
DEFAULT_MODEL_PATH = "/app/workspace/artifacts/best.pt"


# Pydantic models for API
class ScoringRequest(BaseModel):
    """Request model for scoring endpoint"""
    target: dict = Field(..., description="Target signature with genes and weights")
    method: str = Field("baseline", description="Scoring method: baseline or metric")
    top_k: int = Field(50, ge=1, le=1000, description="Number of top results to return")
    cell_line: Optional[str] = Field(None, description="Filter by cell line")
    blend: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor for metric method")
    auto_blend: bool = Field(False, description="Automatically optimize blend factor")


class ScoringResponse(BaseModel):
    """Response model for scoring endpoint"""
    method: str
    ranking: list
    metadata: dict
    execution_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime: float


def _resolve_lincs_path() -> Path:
    """Resolve the LINCS dataset path from environment variables."""
    env_path = os.environ.get("SCPC_LINCS_PATH", DEFAULT_LINCS_PATH)
    return Path(env_path).expanduser()


def _resolve_model_path() -> Path:
    """Resolve the DualEncoder model path from environment variables."""
    env_path = os.environ.get("SCPC_MODEL_PATH", DEFAULT_MODEL_PATH)
    return Path(env_path).expanduser()


@lru_cache(maxsize=1)
def _load_lincs_cached(path_str: str) -> pd.DataFrame:
    """
    Load the LINCS library and cache the in-memory DataFrame.

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
        return table.to_pandas()

    # File-based dataset (Parquet/CSV/TSV) – rely on existing loader.
    return load_lincs_long(str(path))


def get_lincs_library() -> pd.DataFrame:
    """Return the cached LINCS DataFrame, raising HTTP errors on failure."""
    path = _resolve_lincs_path()
    try:
        return _load_lincs_cached(str(path.resolve()))
    except FileNotFoundError as exc:
        logger.error("LINCS library missing: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to load LINCS library from %s", path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load LINCS library: {exc}",
        ) from exc


def get_model_path(required: bool) -> Optional[str]:
    """Return the model path if present; error if required but absent."""
    model_path = _resolve_model_path()
    if not required:
        return str(model_path) if model_path.exists() else None
    if not model_path.exists():
        msg = f"Metric model not found at {model_path}"
        logger.error(msg)
        raise HTTPException(status_code=503, detail=msg)
    return str(model_path)


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("Starting scPerturb-CMap API server...")
    app.state.start_time = time.time()
    app.state.metrics = get_metrics_collector()

    # Warm the LINCS cache to catch issues early (best effort)
    try:
        get_lincs_library()
    except HTTPException:
        # Defer error handling to request time; log for visibility.
        logger.warning("LINCS library warmup failed; will retry on demand.")

    yield
    
    # Shutdown
    print("Shutting down scPerturb-CMap API server...")


# Create FastAPI app
app = FastAPI(
    title="scPerturb-CMap API",
    description="REST API for single-cell connectivity mapping",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    
    # Increment active requests
    if hasattr(app.state, 'metrics'):
        app.state.metrics.active_requests.inc()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    if hasattr(app.state, 'metrics'):
        app.state.metrics.active_requests.dec()
        app.state.metrics.record_http_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=duration
        )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(duration)
    
    return response


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        uptime=uptime
    )


# Readiness probe
@app.get("/ready", tags=["System"])
async def readiness_check():
    """Readiness probe for Kubernetes"""
    # TODO: Check database connectivity, cache availability, etc.
    return {"status": "ready"}


# Metrics endpoint
@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint"""
    # Metrics are exposed via prometheus_client HTTP server
    return {"message": "Metrics available on port 8000"}


# Scoring endpoint
@app.post("/api/score", response_model=ScoringResponse, tags=["Scoring"])
async def score_target(
    request: ScoringRequest,
    background_tasks: BackgroundTasks
):
    """
    Score a target signature against LINCS library
    
    Returns ranked list of compounds with connectivity scores
    """
    start_time = time.time()

    try:
        # Parse target signature
        target = TargetSignature(
            genes=request.target['genes'],
            weights=request.target['weights'],
            metadata=request.target.get('metadata', {})
        )

        library_df = get_lincs_library()
        model_path = get_model_path(required=request.method.lower() == "metric")

        # Perform scoring
        result = rank_drugs(
            target_signature=target,
            library=library_df,
            method=request.method,
            model_path=model_path,
            top_k=request.top_k,
            blend=request.blend,
            auto_blend=request.auto_blend
        )
        
        # Convert to response format
        ranking_json = result.ranking.to_dict(orient='records')
        execution_time = time.time() - start_time
        
        # Record metrics in background
        if hasattr(app.state, 'metrics'):
            background_tasks.add_task(
                app.state.metrics.record_scoring_operation,
                method=request.method,
                cell_line=request.cell_line,
                duration=execution_time,
                success=True
            )
        
        return ScoringResponse(
            method=result.method,
            ranking=ranking_json,
            metadata={
                **result.metadata,
                'cell_line': request.cell_line
            },
            execution_time=execution_time
        )
    
    except Exception as e:
        # Log error and record metrics
        if hasattr(app.state, 'metrics'):
            background_tasks.add_task(
                app.state.metrics.record_scoring_operation,
                method=request.method,
                cell_line=request.cell_line,
                duration=time.time() - start_time,
                success=False
            )
        
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}"
        )


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return {
        "service": "scPerturb-CMap API",
        "version": "0.2.0",
        "docs": "/api/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
