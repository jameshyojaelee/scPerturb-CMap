# Deployment Documentation

This directory contains comprehensive deployment documentation for production environments.

## Contents

- **[Cloud Deployment Guide](CLOUD_DEPLOYMENT.md)** – AWS, GCP, and Kubernetes walkthrough

## Deployment Options

### Cloud Platforms

1. **AWS**
   - ECS/Fargate for containers
   - Lambda for serverless API
   - S3 for LINCS data storage
   - CloudFormation templates live under `deployment/aws/`

2. **Google Cloud Platform**
   - GKE for Kubernetes
   - Cloud Functions for serverless
   - GCS for data storage
   - Deployment Manager templates live under `deployment/gcp/`

3. **Kubernetes (Any Cloud)**
   - Helm charts live under `deployment/helm/scperturb-cmap/`
   - Auto-scaling and monitoring included

### Container Deployment

- **Docker**: see `deployment/docker/`
  - Production Dockerfile with optimizations
  - Docker Compose for local deployment
  - Multi-stage builds with caching

### CI/CD

- **GitHub Actions**: `.github/workflows/`
- **GitLab CI**: `deployment/ci/`

## API hardening and observability

- Copy `.env.example` to `.env` (or your platform secret manager) and set `SCPC_API_KEYS` as JSON label:key pairs along with `SCPC_API_KEY_HEADER` (default `X-API-Key`). Requests without a valid key receive HTTP 401.
- Per-principal throttling is controlled by `SCPC_RATE_LIMIT_PER_MINUTE` and `SCPC_RATE_LIMIT_WINDOW_SECONDS`; outages return HTTP 429 with a `Retry-After` hint.
- Structured JSON logs are emitted when `SCPC_JSON_LOGS=true`, capturing method, path, status, duration, client, and a sanitized principal label (never the secret key).
- Metrics now include the principal label as a low-cardinality dimension on HTTP/scoring counters to align dashboards with authentication context.
- Keep Redis/Postgres readiness checks optional by toggling `SCPC_READINESS_CHECK_REDIS` / `SCPC_READINESS_CHECK_POSTGRES` when those services are absent from your deployment tier.

## Quick Links

- Main repository paths:
  - `deployment/` – deployment manifests and tooling
  - `deployment/docker/` – container assets
  - `deployment/helm/` – Kubernetes charts
  - `deployment/prometheus/` – monitoring configuration

## Support

For deployment issues:
- Check the [troubleshooting notes](CLOUD_DEPLOYMENT.md)
- Open a GitHub issue
- Email: support@scperturb-cmap.org
