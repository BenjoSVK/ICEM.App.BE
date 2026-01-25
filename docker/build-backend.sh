#!/bin/bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?


PROJECT_NAME="ikem-backend"
IMAGE_TAG="ikem-backend"

DOCKER_BUILDKIT=1 \
    docker build \
    -f docker/backend/Dockerfile \
    -t "${IMAGE_TAG}" . || exit $?

