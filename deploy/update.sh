#!/usr/bin/env bash
# Post-merge deploy: pull the newest image and restart. Rollback = set
# IMAGE_TAG=sha-<short> in deploy/.env and re-run.
set -euo pipefail
cd "$(dirname "$0")"
docker compose pull
docker compose up -d
docker compose ps
