#!/usr/bin/env bash
# First-deploy gate (phase-0 spec §5): run the tier-2 suites — golden masters
# and GPU/cv2 bit-exactness, INCLUDING the ~10-min slow full-mode golden —
# inside the image, against the repo bind-mounted at its identical host path.
# pytest is installed ephemerally in the container layer; the shipped image
# stays test-free. Usage: deploy/validate.sh [image-tag]   (default: latest)
# NOTE: the gate runs the CHECKOUT's code under the image's runtime — run it
# from a checkout at the commit the image was built from.
set -euo pipefail
TAG="${1:-latest}"
IMAGE="ghcr.io/koncordantlab/water-droplet-segmentation:${TAG}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${REPO}:${REPO}" \
  -w "${REPO}/backend" \
  "${IMAGE}" \
  bash -c "pip install --no-cache-dir --quiet --target /tmp/ptest pytest \
           && PYTHONPATH=/tmp/ptest python -m pytest -m local"
echo "In-container validation PASSED for ${IMAGE}"
# --target /tmp/ptest: the container runs as the invoking (non-root) uid,
# which cannot write to site-packages; /tmp is writable for any uid.
