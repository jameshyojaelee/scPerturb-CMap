# Cloud Deployment Summary - scPerturb-CMap

## 🎉 Deployment Infrastructure Complete

The scPerturb-CMap platform has been fully cloud-enabled with production-ready infrastructure for AWS, GCP, and Kubernetes deployments.

## 📦 What's Included

### 1. Production Docker Images
- **Multi-stage Dockerfile** (`Dockerfile.prod`) with optimized build layers
- **LINCS data caching** layer for faster cold starts
- **Security hardening**: Non-root user, minimal attack surface
- **Multiple targets**: API server, UI, Worker, with independent scaling
- **Multi-architecture support**: AMD64 and ARM64

### 2. Kubernetes Deployment (Helm Charts)
Location: `deployment/helm/scperturb-cmap/`

**Features:**
- Auto-scaling for API (2-10 pods) and Workers (2-20 pods)
- Horizontal Pod Autoscaler based on CPU and memory
- Persistent volume claims for LINCS data and models
- Service mesh ready with health checks
- Resource limits and requests properly configured
- Redis and PostgreSQL included for caching and metadata

**Quick Deploy:**
```bash
helm install scperturb-cmap ./deployment/helm/scperturb-cmap \
  --set image.repository=YOUR_REGISTRY/scperturb-cmap \
  --set image.tag=0.2.0
```

### 3. AWS Deployment Templates
Location: `deployment/aws/cloudformation/`

**Components:**
- **VPC Networking** (`vpc-networking.yaml`): Multi-AZ VPC with public/private subnets, NAT gateways
- **S3 Storage** (`s3-storage.yaml`): Optimized buckets for LINCS data, models, results with lifecycle policies
- **ECS Fargate** (`ecs-fargate.yaml`): Serverless containers with ALB, auto-scaling, and CloudWatch
- **Lambda API** (`lambda-api.yaml`): Serverless scoring API with API Gateway, 10GB memory, 15-min timeout

**Cost-Optimized:**
- Fargate Spot instances for workers (70% cost savings)
- S3 Intelligent-Tiering for automatic cost optimization
- CloudFront CDN for LINCS data caching

### 4. GCP Deployment Templates
Location: `deployment/gcp/deployment-manager/`

**Components:**
- **GKE Cluster** (`gke-cluster.yaml`): Private GKE with multiple node pools (system, API, workers)
- **GCS Storage** (`gcs-storage.yaml`): Cloud Storage buckets with versioning and lifecycle rules
- **Cloud Functions** (`cloud-functions/`): Serverless scoring with 8GB memory

**Features:**
- Workload Identity for secure GCS access
- Preemptible VMs for cost-efficient workers
- Binary Authorization for security
- Cloud Build integration

### 5. Serverless APIs

#### AWS Lambda
Location: `deployment/aws/lambda/`

**Features:**
- Optimized cold start with lazy imports
- S3 caching of LINCS data in /tmp
- API Gateway with rate limiting and API keys
- CloudWatch metrics and alarms
- Up to 10GB ephemeral storage

**Pricing:** ~$0.20 per 1M requests + $0.0000133/GB-second

#### GCP Cloud Functions
Location: `deployment/gcp/cloud-functions/`

**Features:**
- Gen2 Cloud Functions (Cloud Run-based)
- GCS integration for data loading
- CORS support for web clients
- Automatic scaling to 100+ instances
- 8GB memory, 9-minute timeout

**Pricing:** ~$0.40 per 1M requests + $0.0000025/GB-second

### 6. Cloud-Optimized Data Layer
Location: `src/scperturb_cmap/io/cloud_storage.py`

**Features:**
- **Intelligent partitioning strategies**: By cell_line, compound, or hybrid
- **Predicate pushdown**: Query only needed partitions
- **LRU caching**: Local cache with automatic eviction
- **Multi-cloud support**: Works with S3, GCS, and local filesystems
- **PyArrow integration**: Efficient columnar data access

**Usage:**
```python
from scperturb_cmap.io.cloud_storage import partition_lincs_for_cloud

# Partition LINCS data for cloud
stats = partition_lincs_for_cloud(
    input_path='lincs_level5_long.parquet',
    output_path='s3://bucket/lincs/partitioned',
    strategy='cell_line',
    cloud_provider='aws'
)
# Result: 50+ partitions for efficient cell-line queries
```

### 7. Monitoring and Observability
Location: `deployment/prometheus/`, `deployment/grafana/`

**Prometheus Metrics:**
- HTTP request rates, latency, errors
- Scoring operation duration and success rate
- Resource utilization (CPU, memory, disk)
- Cache hit rates
- Queue lengths and worker status

**Grafana Dashboards:**
- scPerturb-CMap Overview dashboard pre-configured
- Real-time metrics with 15-second refresh
- Alerts for high error rates, resource exhaustion, and downtime

**Cloud-Native Monitoring:**
- CloudWatch integration for AWS (Lambda, ECS)
- Cloud Monitoring integration for GCP
- Custom metrics exported to both platforms

### 8. CI/CD Pipelines
Location: `.github/workflows/`, `.gitlab-ci.yml`

**GitHub Actions Workflow:**
- Automated testing on PRs
- Multi-architecture Docker builds
- Deployment to Kubernetes, Lambda, and Cloud Functions
- Automated rollouts with verification
- Slack notifications

**GitLab CI/CD:**
- Parallel builds for API, UI, and Workers
- Staging and production environments
- Manual approval for production
- Rollback capability
- Coverage reporting

### 9. Production-Ready Features

**Security:**
- Non-root containers with read-only filesystems
- Network policies for pod-to-pod communication
- IAM roles with least-privilege access
- Encryption at rest (S3-SSE, GCS default encryption)
- SSL/TLS termination at load balancer

**Reliability:**
- Health checks and readiness probes
- Pod Disruption Budgets (PDB) for HA
- Multi-AZ deployment for fault tolerance
- Auto-healing with Kubernetes liveness probes
- Circuit breakers and retries

**Performance:**
- Efficient Parquet partitioning (10-100x query speedup)
- CDN caching for LINCS data
- Redis caching for results
- Connection pooling for databases
- Async/await for concurrent operations

**Observability:**
- Structured logging (JSON format)
- Distributed tracing support
- Request ID tracking
- Performance metrics at every layer

## 🚀 Quick Start Guide

### Deploy to AWS ECS (Recommended for Production)
```bash
# 1. Deploy infrastructure
cd deployment/aws/cloudformation
aws cloudformation create-stack \
  --stack-name scperturb-cmap \
  --template-body file://ecs-fargate.yaml \
  --parameters ParameterKey=ImageUri,ParameterValue=YOUR_IMAGE \
  --capabilities CAPABILITY_NAMED_IAM

# 2. Access the application
aws cloudformation describe-stacks \
  --stack-name scperturb-cmap \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue'
```

### Deploy to Kubernetes (Any Cloud)
```bash
# 1. Build and push image
docker build -f Dockerfile.prod -t your-registry/scperturb-cmap:0.2.0 .
docker push your-registry/scperturb-cmap:0.2.0

# 2. Deploy with Helm
helm install scperturb-cmap ./deployment/helm/scperturb-cmap \
  --set image.repository=your-registry/scperturb-cmap \
  --set image.tag=0.2.0 \
  --namespace scperturb-cmap --create-namespace

# 3. Access UI
kubectl port-forward svc/scperturb-cmap-ui 8501:8501 -n scperturb-cmap
```

### Deploy Serverless (AWS Lambda)
```bash
# 1. Package and deploy
cd deployment/aws/lambda
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip . && cd ..
zip -g lambda.zip scoring_handler.py

aws lambda create-function \
  --function-name scperturb-cmap-scoring \
  --runtime python3.10 \
  --handler scoring_handler.lambda_handler \
  --zip-file fileb://lambda.zip \
  --role YOUR_LAMBDA_ROLE_ARN \
  --memory-size 10240 \
  --timeout 900

# 2. Create API Gateway endpoint
# (Or use CloudFormation template for full setup)
```

## 📊 Architecture Overview

```
                                    ┌─────────────────┐
                                    │   Cloud CDN     │
                                    │  (CloudFront/   │
                                    │   Cloud CDN)    │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │                                                  │
         ┌──────────▼──────────┐                          ┌──────────▼──────────┐
         │  Application Load   │                          │    API Gateway      │
         │     Balancer        │                          │  (Serverless API)   │
         └──────────┬──────────┘                          └──────────┬──────────┘
                    │                                                  │
       ┌────────────┴────────────┐                          ┌─────────▼─────────┐
       │                         │                          │  Lambda/Cloud     │
┌──────▼─────┐           ┌──────▼─────┐                   │    Functions      │
│  API Pods  │           │  UI Pods   │                   │   (Serverless)    │
│  (2-10x)   │           │  (1-5x)    │                   └─────────┬─────────┘
└──────┬─────┘           └──────┬─────┘                             │
       │                        │                                    │
       └────────────┬───────────┘                                    │
                    │                                                 │
         ┌──────────▼──────────┐                          ┌─────────▼─────────┐
         │   Redis Cache       │                          │  S3/GCS LINCS     │
         │   (In-Memory)       │                          │  Data (Cached)    │
         └──────────┬──────────┘                          └───────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Worker Pods        │
         │  (2-20x, Autoscale) │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  LINCS Data (S3/GCS)│
         │  Partitioned Parquet│
         └─────────────────────┘
```

## 💰 Cost Estimates

### Serverless (Intermittent Use)
- **100 requests/day**: ~$5/month
- **1,000 requests/day**: ~$30/month
- **10,000 requests/day**: ~$250/month
- Best for: Research, development, batch jobs

### Container-Based (24/7 Production)
- **AWS ECS Fargate**: ~$200-500/month (2 API + 3 workers)
- **GCP GKE**: ~$150-400/month (with preemptible VMs)
- **Data storage (S3/GCS)**: ~$50-100/month for 100GB
- Best for: Production, high-traffic applications

## 📝 Next Steps

1. **Configure secrets**: Set up AWS Secrets Manager or GCP Secret Manager for credentials
2. **Upload LINCS data**: Use the partitioning scripts to upload optimized data
3. **Set up monitoring**: Configure alert webhooks for Slack/PagerDuty
4. **Enable HTTPS**: Configure TLS certificates (Let's Encrypt recommended)
5. **Tune auto-scaling**: Adjust HPA thresholds based on observed load patterns
6. **Set up backups**: Configure automated backups for PostgreSQL and Redis

## 🔗 Additional Resources

- [Deployment Guide](deployment/README.md) - Detailed deployment instructions
- [Helm Chart Documentation](deployment/helm/scperturb-cmap/README.md)
- [API Documentation](docs/api.md)
- [Architecture Diagrams](docs/architecture/) (TODO: Create diagrams)

## 🎓 Training and Support

For questions or issues:
- GitHub Issues: https://github.com/jameslee/scPerturb-CMap/issues
- Email: support@scperturb-cmap.org
- Slack: #scperturb-cmap

---

**Built with ❤️ for the single-cell community**
