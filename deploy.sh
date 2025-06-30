#!/bin/bash

# Set your project variables
PROJECT_ID="your-gcp-project-id"
SERVICE_NAME="flask-sentiment-analysis"
REGION="asia-southeast1"  # Singapore region (closest to Indonesia)

echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME .

echo "Pushing image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 10 \
  --port 8080

echo "Deployment completed!"
echo "Your app should be available at the URL shown above."
