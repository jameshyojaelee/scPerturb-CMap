# Docker Deployment

This directory contains Docker configurations for scPerturb-CMap deployment.

## Files

- **`Dockerfile`** - Basic Docker image for development
- **`Dockerfile.prod`** - Production-optimized multi-stage Dockerfile with:
  - LINCS data caching layer
  - Security hardening (non-root user)
  - Multiple build targets (API, UI, Worker)
  - Optimized layer caching

- **`docker-compose.prod.yml`** - Full production stack including:
  - API server
  - Streamlit UI
  - Worker processes
  - Redis (caching)
  - PostgreSQL (metadata)
  - Nginx (reverse proxy)
  - Prometheus & Grafana (monitoring)

- **`.dockerignore`** - Excludes unnecessary files from Docker build context

## Quick Start

### Development

```bash
# Build image
docker build -f Dockerfile -t scperturb-cmap:dev .

# Run CLI
docker run scperturb-cmap:dev diagnose
```

### Production

```bash
# Build production image
docker build -f Dockerfile.prod --target api-server -t scperturb-cmap:prod .

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## Build Targets

The production Dockerfile has multiple targets:

1. **builder** - Compiles dependencies and builds wheels
2. **lincs-cache** - Optional layer for pre-cached LINCS data
3. **runtime** - Base runtime with Streamlit UI
4. **api-server** - FastAPI REST API
5. **worker** - Celery background workers

Build specific target:
```bash
docker build -f Dockerfile.prod --target api-server -t scperturb-cmap:api .
```

## Environment Variables

### Required
- `SCPC_DATA_DIR` - Data directory path (default: `/data`)
- `SCPC_LINCS` - LINCS library path (default: `/data/lincs/partitioned`)

### Optional
- `SCPC_MODEL` - Model checkpoint path
- `SCPC_CACHE_DIR` - Cache directory
- `REDIS_URL` - Redis connection URL (for workers)
- `DATABASE_URL` - PostgreSQL connection URL

## Mounting Data

```bash
docker run -v /host/data:/data scperturb-cmap:prod
```

## Health Checks

All services include health check endpoints:
- API: `http://localhost:8000/health`
- UI: `http://localhost:8501/_stcore/health`

## See Also

- [Cloud Deployment Guide](../../docs/deployment/CLOUD_DEPLOYMENT.md)
- [Kubernetes Helm Charts](../kubernetes/helm/)
- [Production Deployment](../README.md)
