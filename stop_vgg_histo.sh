#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define container names
REDIS_CONTAINER="vgg_histo_redis"
BACKEND_CONTAINER="vgg_histo_backend"
WORKER_CONTAINER="vgg_histo_celery_worker"
NETWORK_NAME="vgg_histo_network"
DB_CONTAINER="vgg_histo_db"

# Step 1: Stop and remove containers
echo "Stopping and removing containers..."
docker stop "$REDIS_CONTAINER" "$BACKEND_CONTAINER" "$WORKER_CONTAINER" "$DB_CONTAINER" 2>/dev/null || true
docker rm "$REDIS_CONTAINER" "$BACKEND_CONTAINER" "$WORKER_CONTAINER" "$DB_CONTAINER" 2>/dev/null || true

# Step 2: Remove the Docker network (if it exists)
echo "Removing Docker network: $NETWORK_NAME..."
if docker network ls | grep -q "$NETWORK_NAME"; then
  docker network rm "$NETWORK_NAME"
else
  echo "Docker network $NETWORK_NAME does not exist. Skipping removal."
fi

echo "All containers stopped and resources cleaned up!"