#!/usr/bin/env bash
# Build the Docker image and run verification.
set -euo pipefail

IMAGE_NAME="${1:-runpod-video-processor}"
TAG="${2:-latest}"

echo "Building ${IMAGE_NAME}:${TAG} ..."
docker build -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Image built successfully."
echo "Size: $(docker image inspect "${IMAGE_NAME}:${TAG}" --format='{{.Size}}' | numfmt --to=iec-i 2>/dev/null || docker image inspect "${IMAGE_NAME}:${TAG}" --format='{{.Size}}')"
echo ""
echo "To run locally:"
echo "  docker-compose up"
echo ""
echo "To push to Docker Hub:"
echo "  docker tag ${IMAGE_NAME}:${TAG} your-dockerhub-user/${IMAGE_NAME}:${TAG}"
echo "  docker push your-dockerhub-user/${IMAGE_NAME}:${TAG}"
