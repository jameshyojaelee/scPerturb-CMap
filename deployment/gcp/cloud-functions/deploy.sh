#!/bin/bash
# Deploy scPerturb-CMap scoring Cloud Function to GCP

set -e

PROJECT_ID=${GCP_PROJECT:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
LINCS_BUCKET=${LINCS_BUCKET:-"scperturb-cmap-lincs-data"}
MODEL_BUCKET=${MODEL_BUCKET:-"scperturb-cmap-models"}

echo "Deploying scPerturb-CMap Cloud Functions to project: $PROJECT_ID"

# Deploy scoring function
echo "Deploying scoring function..."
gcloud functions deploy scperturb-cmap-score \
    --gen2 \
    --runtime=python310 \
    --region=$REGION \
    --source=. \
    --entry-point=score \
    --trigger-http \
    --allow-unauthenticated \
    --memory=8GB \
    --timeout=540s \
    --max-instances=100 \
    --min-instances=0 \
    --set-env-vars="GCP_PROJECT=$PROJECT_ID,LINCS_BUCKET=$LINCS_BUCKET,MODEL_BUCKET=$MODEL_BUCKET" \
    --service-account=scperturb-cmap-app@$PROJECT_ID.iam.gserviceaccount.com

# Deploy health check function
echo "Deploying health check function..."
gcloud functions deploy scperturb-cmap-health \
    --gen2 \
    --runtime=python310 \
    --region=$REGION \
    --source=. \
    --entry-point=health \
    --trigger-http \
    --allow-unauthenticated \
    --memory=256MB \
    --timeout=10s \
    --max-instances=10 \
    --min-instances=0

echo "Deployment complete!"
echo ""
echo "Scoring endpoint:"
gcloud functions describe scperturb-cmap-score --region=$REGION --gen2 --format="value(serviceConfig.uri)"
echo ""
echo "Health endpoint:"
gcloud functions describe scperturb-cmap-health --region=$REGION --gen2 --format="value(serviceConfig.uri)"
