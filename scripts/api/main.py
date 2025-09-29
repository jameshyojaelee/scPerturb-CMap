"""
FastAPI-based REST API for scPerturb-CMap
Production-ready API server with metrics, health checks, and async support
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import time

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.io.schemas import TargetSignature, ScoreResult
from scperturb_cmap.utils.metrics import get_metrics_collector


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


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("Starting scPerturb-CMap API server...")
    app.state.start_time = time.time()
    app.state.metrics = get_metrics_collector()
    
    # Load shared resources (cache LINCS library if needed)
    # TODO: Implement caching strategy
    
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
        
        # Load LINCS library (TODO: implement caching)
        # For now, assuming library is loaded or passed in
        # In production, use cloud_storage module to load from S3/GCS
        library_path = "/data/lincs/partitioned"  # Configure via environment
        
        # Perform scoring
        result = rank_drugs(
            target_signature=target,
            library=library_path,  # This should be a DataFrame or path
            method=request.method,
            model_path="/app/workspace/artifacts/best.pt" if request.method == "metric" else None,
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
