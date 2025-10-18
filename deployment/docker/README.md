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

The API and worker containers read their configuration from environment variables. Defaults are
optimised for local development; production deployments should override them explicitly.

### Core runtime
- `SCPC_ENV` – Deployment environment (`production`, `staging`, or `development`; default: `production`)
- `SCPC_LINCS_PATH` – Absolute path to the LINCS dataset (default: `/data/lincs/partitioned`)
- `SCPC_MODEL_PATH` – Path to the metric model checkpoint (default: `/app/workspace/artifacts/best.pt`)
- `SCPC_CACHE_TTL` – LINCS cache time-to-live in seconds (default: `3600`)
- `SCPC_REQUEST_TIMEOUT` – Request timeout enforced by the API in seconds (default: `30`)
- `SCPC_MAX_REQUEST_SIZE_MB` – Maximum accepted request payload size in MiB (default: `25`)

### Networking & security
- `SCPC_CORS_ORIGINS` – Comma-separated or JSON list of allowed CORS origins; falls back to `*` only when `SCPC_ENV=development`
- `SCPC_REQUIRE_MODEL` – Set to `true` to make readiness checks fail when the model file is missing (default: `false`)

### Metrics & observability
- `SCPC_METRICS_BACKEND` – Metrics backend (`prometheus`, `cloudwatch`, `cloud_monitoring`, or `none`; default: `prometheus`)
- `SCPC_METRICS_PORT` – Port for the Prometheus metrics HTTP exporter (default: `8000`)
- `SCPC_METRICS_NAMESPACE` – Optional namespace for CloudWatch or Cloud Monitoring metrics

### External services
- `SCPC_REDIS_URL` or `REDIS_URL` – Redis connection string for caching (optional)
- `SCPC_DATABASE_URL` or `DATABASE_URL` – PostgreSQL connection string for metadata (optional)
- `SCPC_READINESS_CHECK_REDIS` / `SCPC_READINESS_CHECK_POSTGRES` – Enable/disable readiness probes for Redis/PostgreSQL (default: `true`)

### Data directories
- `SCPC_DATA_DIR` – Shared data directory path (default: `/data`)
- `SCPC_CACHE_DIR` – Writeable cache directory (default: `/app/workspace/cache`)

> **Security note:** API authentication, rate limiting, and key management are still handled by the
> ingress or API gateway tiers. Ensure these are configured before exposing the service publicly.

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
