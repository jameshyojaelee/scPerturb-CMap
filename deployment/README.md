# scPerturb-CMap Cloud Deployment Guide

This directory contains comprehensive cloud deployment configurations for scPerturb-CMap, enabling production-ready deployments on AWS, GCP, and Kubernetes platforms.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
  - [Kubernetes (Helm)](#kubernetes-helm)
  - [AWS ECS/Fargate](#aws-ecsfargate)
  - [AWS Lambda (Serverless)](#aws-lambda-serverless)
  - [GCP GKE](#gcp-gke)
  - [GCP Cloud Functions (Serverless)](#gcp-cloud-functions-serverless)
- [Data Management](#data-management)
- [Monitoring and Logging](#monitoring-and-logging)
- [CI/CD Integration](#cicd-integration)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

The deployment architecture supports:
- **Container orchestration**: Kubernetes (EKS, GKE, self-hosted)
- **Serverless APIs**: AWS Lambda, GCP Cloud Functions
- **Auto-scaling**: HPA for containers, Lambda/Cloud Functions concurrency
- **Data storage**: S3/GCS with optimized Parquet partitioning
- **Monitoring**: Prometheus, Grafana, CloudWatch, Cloud Monitoring
- **Security**: IAM roles, network policies, encryption at rest/transit

## Prerequisites

### General
- Docker 20.10+
- kubectl 1.24+
- Helm 3.10+

### AWS
- AWS CLI v2
- Configured AWS credentials (`~/.aws/credentials`)
- Permissions for CloudFormation, ECS, Lambda, S3

### GCP
- gcloud CLI
- Authenticated GCP account (`gcloud auth login`)
- Enabled APIs: GKE, GCS, Cloud Functions, Cloud Build

## Quick Start

### 1. Build Production Docker Image

```bash
# Build multi-stage production image
docker build -f Dockerfile.prod -t scperturb-cmap:prod .

# Tag for your registry
docker tag scperturb-cmap:prod YOUR_REGISTRY/scperturb-cmap:0.2.0
docker push YOUR_REGISTRY/scperturb-cmap:0.2.0
```

### 2. Deploy to Kubernetes (Local Testing)

```bash
cd deployment/helm/scperturb-cmap

# Install with defaults
helm install scperturb-cmap . \
  --set image.repository=YOUR_REGISTRY/scperturb-cmap \
  --set image.tag=0.2.0

# Access the UI
kubectl port-forward svc/scperturb-cmap-ui 8501:8501
```

### 3. Deploy Serverless (AWS Lambda)

```bash
# Deploy infrastructure
cd deployment/aws/cloudformation
aws cloudformation deploy \
  --template-file vpc-networking.yaml \
  --stack-name scperturb-cmap-vpc

aws cloudformation deploy \
  --template-file s3-storage.yaml \
  --stack-name scperturb-cmap-storage

# Deploy Lambda function
cd ../lambda
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip . && cd ..
zip -g lambda.zip scoring_handler.py

aws lambda create-function \
  --function-name scperturb-cmap-scoring \
  --runtime python3.10 \
  --handler scoring_handler.lambda_handler \
  --zip-file fileb://lambda.zip \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role
```

## Deployment Options

### Kubernetes (Helm)

Full production deployment with auto-scaling, monitoring, and high availability.

**Configuration:**
```yaml
# values.yaml customization
api:
  autoscaling:
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

persistence:
  lincsData:
    size: 100Gi
    existingClaim: "lincs-data-pvc"
```

**Deploy:**
```bash
helm install scperturb-cmap ./deployment/helm/scperturb-cmap \
  -f custom-values.yaml \
  --namespace scperturb-cmap \
  --create-namespace
```

**Access:**
```bash
# Get LoadBalancer IP
kubectl get svc scperturb-cmap-ui -n scperturb-cmap

# Or use port-forward for testing
kubectl port-forward svc/scperturb-cmap-ui 8501:8501 -n scperturb-cmap
```

### AWS ECS/Fargate

Fully managed container service without Kubernetes complexity.

**Deploy:**
```bash
cd deployment/aws/cloudformation

# 1. VPC and networking
aws cloudformation deploy \
  --template-file vpc-networking.yaml \
  --stack-name scperturb-cmap-vpc \
  --parameter-overrides EnvironmentName=scperturb-cmap

# 2. S3 storage
aws cloudformation deploy \
  --template-file s3-storage.yaml \
  --stack-name scperturb-cmap-storage \
  --capabilities CAPABILITY_NAMED_IAM

# 3. ECS cluster and services
aws cloudformation deploy \
  --template-file ecs-fargate.yaml \
  --stack-name scperturb-cmap-ecs \
  --parameter-overrides ImageUri=YOUR_ECR_IMAGE \
  --capabilities CAPABILITY_NAMED_IAM
```

**Access:**
```bash
# Get ALB DNS name
aws cloudformation describe-stacks \
  --stack-name scperturb-cmap-ecs \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
  --output text
```

### AWS Lambda (Serverless)

Cost-effective serverless deployment for intermittent workloads.

**Features:**
- Pay-per-invocation pricing
- Auto-scaling to 1000+ concurrent executions
- 15-minute timeout for long-running scoring
- S3 integration for LINCS data caching

**Deploy:**
```bash
cd deployment/aws/cloudformation

# Deploy Lambda stack
aws cloudformation deploy \
  --template-file lambda-api.yaml \
  --stack-name scperturb-cmap-lambda \
  --capabilities CAPABILITY_IAM

# Get API endpoint
aws cloudformation describe-stacks \
  --stack-name scperturb-cmap-lambda \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
  --output text
```

**Usage:**
```bash
# Get API key
aws apigateway get-api-keys --include-values

# Make scoring request
curl -X POST https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/score \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @request.json
```

### GCP GKE

Google Kubernetes Engine deployment with auto-scaling and Workload Identity.

**Deploy:**
```bash
cd deployment/gcp/deployment-manager

# 1. Create GCS storage
gcloud deployment-manager deployments create scperturb-storage \
  --config gcs-storage.yaml

# 2. Create GKE cluster
gcloud deployment-manager deployments create scperturb-cluster \
  --config gke-cluster.yaml

# 3. Deploy Helm chart
gcloud container clusters get-credentials scperturb-cmap-cluster \
  --zone us-central1-a

helm install scperturb-cmap ../helm/scperturb-cmap
```

### GCP Cloud Functions (Serverless)

Serverless deployment on Google Cloud Platform.

**Deploy:**
```bash
cd deployment/gcp/cloud-functions

# Set environment variables
export GCP_PROJECT="your-project-id"
export LINCS_BUCKET="scperturb-cmap-lincs-data"
export MODEL_BUCKET="scperturb-cmap-models"

# Deploy functions
bash deploy.sh
```

**Usage:**
```bash
# Get function URL
gcloud functions describe scperturb-cmap-score --gen2 --region=us-central1 \
  --format="value(serviceConfig.uri)"

# Make request
curl -X POST https://YOUR_FUNCTION_URL/score \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Data Management

### Cloud-Optimized Parquet Partitioning

Optimize LINCS data for cloud storage with efficient partitioning:

```python
from scperturb_cmap.io.cloud_storage import partition_lincs_for_cloud

# Partition by cell line for efficient filtering
stats = partition_lincs_for_cloud(
    input_path='lincs_level5_long.parquet',
    output_path='s3://bucket/lincs/partitioned',
    strategy='cell_line',
    cloud_provider='aws'
)

print(f"Created {stats['num_partitions']} partitions")
```

**Partitioning Strategies:**
- `cell_line`: Best for cell-line-specific queries (most common)
- `compound`: Best for compound-specific lookups
- `cell_line_compound`: Hybrid for both query patterns
- `date`: For temporal analysis

### Upload LINCS Data

**AWS S3:**
```bash
aws s3 sync ./data/lincs/ s3://scperturb-cmap-lincs-data/lincs/ \
  --storage-class INTELLIGENT_TIERING
```

**GCP GCS:**
```bash
gsutil -m rsync -r ./data/lincs/ gs://scperturb-cmap-lincs-data/lincs/
```

## Monitoring and Logging

### Prometheus + Grafana

Included monitoring stack with pre-configured dashboards.

**Access Grafana:**
```bash
# Port-forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n scperturb-cmap

# Default credentials: admin/admin
open http://localhost:3000
```

**Key Dashboards:**
- **scPerturb-CMap Overview**: Request rates, latency, errors
- **Resource Usage**: CPU, memory, disk utilization
- **Worker Queue**: Celery task metrics

### CloudWatch (AWS)

Lambda and ECS metrics automatically published to CloudWatch.

**View Logs:**
```bash
# Lambda logs
aws logs tail /aws/lambda/scperturb-cmap-scoring --follow

# ECS logs
aws logs tail /ecs/scperturb-cmap --follow
```

### Cloud Monitoring (GCP)

**View Metrics:**
```bash
gcloud monitoring dashboards list
```

## CI/CD Integration

### GitHub Actions

See `.github/workflows/deploy.yml` for automated deployments.

**Manual Trigger:**
```bash
gh workflow run deploy.yml \
  -f environment=production \
  -f image_tag=v0.2.0
```

### GitLab CI/CD

See `.gitlab-ci.yml` for pipeline configuration.

## Cost Optimization

### Serverless (Lowest Cost for Intermittent Use)
- **AWS Lambda**: ~$0.20 per 1M requests + compute time
- **No idle costs** when not in use
- Best for: Research workloads, batch processing

### Container-based (Predictable Costs)
- **ECS Fargate**: ~$0.04/vCPU/hour + $0.004/GB/hour
- **GKE**: ~$0.10/hour per node (preemptible: ~$0.015/hour)
- Best for: Production, 24/7 availability

### Cost-Saving Tips
1. Use **spot/preemptible instances** for workers (60-90% savings)
2. Enable **auto-scaling** to scale down during low traffic
3. Use **S3 Intelligent-Tiering** or **GCS Autoclass** for data
4. Set **CloudFront/CDN** for LINCS data caching
5. Configure **lifecycle policies** to archive old results

## Troubleshooting

### Common Issues

**1. Pod/Container Won't Start**
```bash
# Check logs
kubectl logs -l app.kubernetes.io/component=api -n scperturb-cmap

# Check events
kubectl get events -n scperturb-cmap --sort-by='.lastTimestamp'
```

**2. Lambda Timeout**
- Increase timeout (max 15 minutes)
- Optimize LINCS data size (use partitioning)
- Increase memory allocation (faster CPU)

**3. Out of Memory**
- Increase pod memory limits
- Check for memory leaks in logs
- Optimize data loading (use columnar reads)

**4. Slow Scoring**
- Verify LINCS data is partitioned
- Check S3/GCS network latency
- Enable CloudFront/CDN caching

### Health Checks

```bash
# Kubernetes
kubectl get pods -n scperturb-cmap

# ECS
aws ecs describe-services \
  --cluster scperturb-cmap-cluster \
  --services scperturb-cmap-api

# Lambda
aws lambda invoke --function-name scperturb-cmap-health /dev/stdout
```

## Additional Resources

- [Architecture Diagrams](./docs/architecture.md)
- [Security Best Practices](./docs/security.md)
- [Performance Tuning](./docs/performance.md)
- [Backup and Recovery](./docs/backup.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/jameslee/scPerturb-CMap/issues
- Documentation: https://scperturb-cmap.readthedocs.io
- Email: support@scperturb-cmap.org
