#!/bin/bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?


PROJECT_NAME="ikem-worker"
IMAGE_TAG="ikem-worker"

DOCKER_BUILDKIT=1 \
    docker build \
    -f docker/worker/Dockerfile \
    -t "${IMAGE_TAG}" . || exit $?

