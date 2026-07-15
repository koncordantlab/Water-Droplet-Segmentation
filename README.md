# Water Droplet — Full Stack Project

This repository contains the full stack Water Droplet project. It includes a Python backend (a Flask server, the `droplet_backend` package) in `backend/` and a React frontend in `frontend/`.

There are two ways to run it: **Docker** (recommended — one container serving both the UI and the API, no Python or npm setup on the host) or the **manual development setup** (backend from source, frontend built or on its own dev server).

## Running with Docker (recommended)

Merging to `main` builds and publishes `ghcr.io/koncordantlab/water-droplet-segmentation`
(`:latest` + a per-merge `:sha-<short>`) via `.github/workflows/release.yml`.
The image contains the API, the built React UI, and CUDA torch — **never the
weights or any videos** (`.dockerignore` enforces this; they are volume mounts).

One-time host prerequisites: Docker with the NVIDIA Container Toolkit
(`nvidia-ctk runtime configure --runtime=docker`).

First deploy on the box, in order:

1. Free port 8050 — stop any manually run `python -m droplet_backend` still running.
2. `cp deploy/.env.example deploy/.env` and fill in the box's paths and uid/gid.
3. Make the GHCR package public in its settings — or `docker login ghcr.io` with a `read:packages` token.
4. `deploy/update.sh` — pulls `:latest`, starts the stack, and waits for it to report healthy.
5. `deploy/validate.sh latest` — runs the golden-master + GPU bit-exactness suites (including the ~10-min full-mode golden) inside the container. Run it from a checkout at the commit the image was built from: the gate exercises the checkout's code under the image's runtime.
6. Rollback drill: set `IMAGE_TAG=sha-<short>` (the first release's tag) in `deploy/.env`, re-run `deploy/update.sh`, then set it back to `latest`.

After that, deploying a merge is just `deploy/update.sh`; rolling back is setting
`IMAGE_TAG=sha-<short>` in `deploy/.env` and re-running it.

The app serves on `127.0.0.1:8050` (UI at `/`, API under `/api/*`). Videos are
readable from `VIDEO_ROOT` (mounted at its identical host path, so paths pasted
into the UI keep working and outputs land next to inputs, exactly like a manual
run).

## Running manually (development setup)

The from-source setup for development: the backend runs from a Python
environment, and the React UI is either built once and served by the backend or
run on its own hot-reload dev server.

### Backend

1. Change to the backend folder:
```bash
cd backend
```

2. Create and activate a virtual environment (recommended):

```bash
# macOS / Linux (zsh)
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1
```

3. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

4. Run the backend:

```bash
python -m droplet_backend
```

The backend will open a server on port 8050 by default. Point your browser to:

http://localhost:8050

All environment variables are optional (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `DROPLET_HOST` | `127.0.0.1` | Flask bind host. |
| `DROPLET_PORT` | `8050` | Flask bind port. |
| `DROPLET_WEIGHTS_PATH` | `backend/app_root/weights_DP(8).pt` | YOLO weights file. Loaded lazily on the first `/api/process` call, not at boot. |
| `DROPLET_WEBUI_DIR` | auto-detect: `droplet_backend/webui/`, then `../frontend/build/` | Directory with the built React `index.html` + static assets. If none is found, `/` serves a plain "UI not built" page while `/api/*` keeps working. |
| `DROPLET_VIDEO_ROOTS` | the invoking user's home directory | Colon-separated allowlist of directories `/api/process` may read `video_path` from; a path outside every root (after resolving symlinks) is rejected with `403`. |
| `DROPLET_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins allowed to call `/api/*` with credentials. A literal `*` makes the server refuse to start, since reflected origins plus credentials would defeat the check. |

### Frontend

Install packages first:

```bash
cd frontend
npm install
```

Then pick one of two modes:

**Build once, served by the backend** (mirrors the Docker deployment — no configuration needed, the frontend's default relative `/api` base works same-origin):

```bash
npm run build
```

The backend auto-detects `frontend/build/` and serves the UI at http://localhost:8050/.

**Hot-reload dev server** (for UI work). It runs on its own port, so it needs a `.env` in `frontend/` pointing at the backend:

```
# frontend/.env
REACT_APP_BACKEND_API_URL=http://localhost:8050/api
```

```bash
npm start
```

This opens the React app on http://localhost:3000, calling the backend at the URL from `.env`.

## Running backend tests

From `backend/`, using the project's Python environment:

```bash
python -m pytest -m "not local"           # tier 1: CPU-only, no weights/GPU needed (this is what CI runs)
python -m pytest -m "local and not slow"  # tier 2: GPU + weights; required before opening a PR
python -m pytest -m "local and slow"      # full-mode golden (~10 min); run before merging numeric changes
```

Tier 1 exercises the pure-Python/numpy modules plus the API/SSE contract against a fake model (no weights or GPU). Tier 2 loads the real weights and runs the two fast basic-mode golden masters; the `slow` marker adds the full-mode golden. Golden masters live in `backend/tests/golden/expected/*.json` — re-record them with `python tests/golden/record_goldens.py` only when a numeric change is intended and reviewed.

Run the tier-2/golden suites (and `deploy/validate.sh`) on an **idle GPU**: concurrent training or other GPU jobs on the same machine make inference nondeterministic under load, so golden comparisons can flake even when the code is correct. If a golden fails, check `nvidia-smi` for other GPU processes before suspecting a regression.

## Tracking pipeline

Standalone research tooling (separate from the served app) in `backend/tracking/`:
a two-phase pipeline — YOLO detection once into `detections.json`, then a custom
tracker (with merge detection) into `tracking_log.json` + an annotated video —
plus analysis/plotting consumers. Run from `backend/`:

```bash
python -m tracking all                      # detect + track with the config.py defaults
python -m tracking detect --video V --detections D
python -m tracking track  --video V --detections D --output OUT.mp4 --log LOG.json
python -m tracking analyze --detections D --tracking-log LOG.json   # insights JSON + CSVs
python -m tracking plots   --tracking-log LOG.json   # timeline plots
```

Tuning lives in `backend/tracking/config.py` (~110 documented tuning constants) —
prefer adjusting constants over editing the matchers, which are heavily
interdependent. Uses `weights_DP(6).pt` on purpose (the thresholds were tuned
against it). The Docker image does not include this package.

## Development instructions & best practices

- Do NOT commit large generated results such as Excel summaries, segmentation videos, model weight checkpoints, or large dataset files. Keep these local or store them in an external storage bucket if needed.
- Work on a feature branch and not in the "main" branch. And make pull request to be reviewed by your peers before merging.
- When adding files to commit, select only the files you intend to push. Avoid using:

```bash
# DO NOT run
git add .
```

Instead, add specific files or directories explicitly, for example:

```bash
git add frontend/src/components/MyComponent.js
git add backend/droplet_backend/api/routes.py
```

- Useful git workflow tips:
	- Use `git status` to review changed files before adding.
	- Use `git add -p` to stage hunks interactively when appropriate.
	- If you accidentally staged large files, use `git reset <file>` to unstage them before committing.

## Files/paths you should NOT push

- Any generated segmentation videos or exports (e.g., `segmentation results/`)
- Local environment folders and virtualenvs (`.venv/`, `venv/`, `env/`)

Add appropriate patterns to your `.gitignore` to keep these out of the repo (if not already present).
