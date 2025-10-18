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
