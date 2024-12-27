#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define container names
REDIS_CONTAINER="vgg_histo_redis"
BACKEND_CONTAINER="vgg_histo_backend"
WORKER_CONTAINER="vgg_histo_celery_worker"
NETWORK_NAME="vgg_histo_network"
DB_CONTAINER="vgg_histo_db"

# Step 1: Build the images
echo "Building Docker images..."
if [[ "$1" == "--build" ]]; then
    docker-compose -f docker-compose.prod.yaml build
else
    echo "Skipping Docker image build. Use --build to build the images."
fi

# Step 2: Create the network if it doesn't exist
if ! docker network ls | grep -q "$NETWORK_NAME"; then
  echo "Creating Docker network: $NETWORK_NAME..."
  docker network create "$NETWORK_NAME"
else
  echo "Docker network $NETWORK_NAME already exists."
fi

# Step 3: Remove existing containers if they exist
echo "Cleaning up existing containers..."
docker rm -f "$REDIS_CONTAINER" 2>/dev/null || true
docker rm -f "$BACKEND_CONTAINER" 2>/dev/null || true
docker rm -f "$WORKER_CONTAINER" 2>/dev/null || true
docker rm -f "$DB_CONTAINER" 2>/dev/null || true

# Step 4: Run Redis container
echo "Starting Redis container..."
docker run -d \
  --name "$REDIS_CONTAINER" \
  --network "$NETWORK_NAME" \
  redis

echo "Starting Database container..."
docker run -d \
  --name "$DB_CONTAINER" \
  --network "$NETWORK_NAME" \
  -e POSTGRES_DB=iedl_db \
  -e POSTGRES_USER=iedl_user \
  -e POSTGRES_PASSWORD=iedl_password \
  -v db_data:/var/lib/postgresql/data \
  postgres:13

# Step 5: Run Backend container
echo "Starting Backend container..."
docker run -d \
  --name "$BACKEND_CONTAINER" \
  -p 7030:8000 \
  --privileged \
  --gpus all \
  -v $(pwd)/src:/app \
  -v /mnt/persist/xtanczost/iedl_root_dir:/iedl_root_dir \
  --env-file .env.prod \
  --network "$NETWORK_NAME" \
  icemappbe_vgg_histo_backend

# Step 6: Run Worker container
echo "Starting Worker container..."
docker run -d \
  --name "$WORKER_CONTAINER" \
  --privileged \
  --gpus all \
  -v $(pwd)/src:/app \
  -v /mnt/persist/xtanczost/iedl_root_dir:/iedl_root_dir \
  --env-file .env.prod \
  --network "$NETWORK_NAME" \
  icemappbe_vgg_histo_celery_worker \
  celery -A celery_tasks.process_folder worker --pool=solo --loglevel=info --concurrency=1

# Step 7: Verify running containers
echo "Checking running containers..."
docker ps

# Step 8: Print success message
echo "All containers are up and running!"
