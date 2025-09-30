# Deployment Documentation

This directory contains comprehensive deployment documentation for production environments.

## Contents

- **[CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)** - Complete cloud deployment guide covering AWS, GCP, and Kubernetes

## Deployment Options

### Cloud Platforms

1. **AWS**
   - ECS/Fargate for containers
   - Lambda for serverless API
   - S3 for LINCS data storage
   - CloudFormation templates: `../../deployment/aws/`

2. **Google Cloud Platform**
   - GKE for Kubernetes
   - Cloud Functions for serverless
   - GCS for data storage
   - Deployment Manager templates: `../../deployment/gcp/`

3. **Kubernetes (Any Cloud)**
   - Helm charts: `../../deployment/kubernetes/helm/`
   - Auto-scaling and monitoring included

### Container Deployment

- **Docker**: See `../../deployment/docker/`
  - Production Dockerfile with optimizations
  - Docker Compose for local deployment
  - Multi-stage builds with caching

### CI/CD

- **GitHub Actions**: `.github/workflows/`
- **GitLab CI**: `../../deployment/ci/`

## Quick Links

- [Deployment README](../../deployment/README.md) - Main deployment directory
- [Docker Files](../../deployment/docker/)
- [Helm Charts](../../deployment/kubernetes/helm/)
- [Monitoring](../../deployment/prometheus/)

## Support

For deployment issues:
- Check [troubleshooting guide](CLOUD_DEPLOYMENT.md#troubleshooting)
- Open a GitHub issue
- Email: support@scperturb-cmap.org
