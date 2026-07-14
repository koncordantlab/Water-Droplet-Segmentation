# syntax=docker/dockerfile:1
# Stage 1 — build the React bundle (same Node major as ci.yml)
FROM node:20-slim AS webui
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2 — runtime. Pinned to the env the golden masters were recorded on
# (droplets conda env): Python 3.13, torch 2.9.0+cu128, opencv-python 4.10.0.84.
# Full opencv (not headless) so the wheel is byte-identical to the host env;
# libgl1/libglib2.0-0 are its runtime libs (same fix ci.yml uses).
FROM python:3.13-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./
# CUDA 12.8 wheels first (RTX 5090 / sm_120); requirements.txt's torch==2.9.0
# is then already satisfied by 2.9.0+cu128 (PEP 440 local-version match).
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt
COPY backend/droplet_backend/ ./droplet_backend/
COPY --from=webui /frontend/build/ ./droplet_backend/webui/
# Pre-create the config dirs the ENV block points at (any-uid writable);
# without them ultralytics falls back to another path with a startup warning.
RUN mkdir -m 1777 -p /tmp/mpl /tmp/ultralytics
# 0.0.0.0: the container port is published (host-side binding decides exposure).
# MPLCONFIGDIR/YOLO_CONFIG_DIR: writable under any --user uid (compose runs the
# container as the invoking lab user so outputs land user-owned on the mounts).
ENV DROPLET_HOST=0.0.0.0 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    YOLO_CONFIG_DIR=/tmp/ultralytics
EXPOSE 8050
CMD ["python", "-m", "droplet_backend"]
