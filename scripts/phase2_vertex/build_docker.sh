#!/bin/bash
# Builds and pushes the Docker container to Google Artifact Registry

PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
REPO_NAME="rlhf-repo"
IMAGE_NAME="rlhf-trainer"
TAG="latest"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: $IMAGE_URI"
# We run this from the root of the project
docker build -t $IMAGE_URI -f ../../Dockerfile ../../

echo "Pushing Docker image to Artifact Registry..."
# Ensure you are authenticated: gcloud auth configure-docker us-central1-docker.pkg.dev
docker push $IMAGE_URI

echo "Push complete! Image URI:"
echo $IMAGE_URI
