#!/bin/bash
# Запускать на RunPod Pod с достаточным диском (100GB+).
# Требует: docker login уже выполнен (Docker Hub).

set -e

DOCKER_USER=${DOCKER_USER:-vdinya7}

echo "=== Building longcat-video ==="
docker build -t $DOCKER_USER/longcat-video:latest ./longcat-video
docker push $DOCKER_USER/longcat-video:latest

echo "=== Building longcat-avatar ==="
docker build -t $DOCKER_USER/longcat-avatar:latest ./longcat-avatar
docker push $DOCKER_USER/longcat-avatar:latest

echo "=== Building ltx-video ==="
docker build -t $DOCKER_USER/ltx-video:latest ./ltx-video
docker push $DOCKER_USER/ltx-video:latest

echo "=== Done ==="
