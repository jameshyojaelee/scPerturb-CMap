# CI/CD Configuration

This directory contains continuous integration and deployment configurations.

## Files

- **`.gitlab-ci.yml`** - GitLab CI/CD pipeline configuration

## GitHub Actions

GitHub Actions workflows are located in `.github/workflows/`:
- `build-and-deploy.yml` - Main CI/CD pipeline
- Automated testing, linting, and deployment

## GitLab CI/CD

The GitLab pipeline (`.gitlab-ci.yml`) includes:

### Stages

1. **test** - Lint and run test suite
2. **build** - Build Docker images (API, UI, Worker)
3. **deploy** - Deploy to staging/production
4. **verify** - Health checks and smoke tests

### Jobs

- `lint` - Code quality checks with ruff
- `test` - pytest with coverage reporting
- `build:api`, `build:ui`, `build:worker` - Docker image builds
- `deploy:staging` - Auto-deploy to staging
- `deploy:production` - Manual deploy to production
- `verify:health-check` - Endpoint validation
- `verify:smoke-test` - Basic functionality tests

## Environment Variables (CI/CD)

Set these in your CI/CD environment:

### Docker Registry
- `CI_REGISTRY_USER` - Registry username
- `CI_REGISTRY_PASSWORD` - Registry password

### Kubernetes (if deploying)
- `KUBE_CONFIG` - Kubernetes config
- `KUBE_NAMESPACE` - Target namespace

### AWS (if using)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

### GCP (if using)
- `GCP_SA_KEY` - Service account key
- `GCP_PROJECT_ID`

## Triggering Deployments

### GitLab
- **Staging**: Auto-deploy on push to `develop` branch
- **Production**: Manual deployment from `main` branch

### GitHub Actions
- **Staging**: Auto-deploy on PR merge to `develop`
- **Production**: Manual workflow dispatch or tag push

## See Also

- [GitHub Actions Workflows](../../.github/workflows/)
- [Deployment Guide](../../docs/deployment/)
- [Docker Configuration](../docker/)
