# Water Droplet — Full Stack Project

This repository contains the full stack Water Droplet project. It includes a Python backend (a Flask server, the `nasa_backend` package) in `Nasa_Backend/` and a React frontend in `nasa-frontend/`.

## Running the backend

1. Change to the backend folder:
```bash
cd Nasa_Backend
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
python -m nasa_backend
```

The backend will open a server on port 8050 by default. Point your browser to:

http://localhost:8050

All environment variables are optional (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `NASA_HOST` | `127.0.0.1` | Flask bind host. |
| `NASA_PORT` | `8050` | Flask bind port. |
| `NASA_WEIGHTS_PATH` | `Nasa_Backend/app_root/weights_DP(8).pt` | YOLO weights file. Loaded lazily on the first `/api/process` call, not at boot. |
| `NASA_WEBUI_DIR` | auto-detect: `nasa_backend/webui/`, then `../nasa-frontend/build/` | Directory with the built React `index.html` + static assets. If none is found, `/` serves a plain "UI not built" page while `/api/*` keeps working. |
| `NASA_VIDEO_ROOTS` | the invoking user's home directory | Colon-separated allowlist of directories `/api/process` may read `video_path` from; a path outside every root (after resolving symlinks) is rejected with `403`. |
| `NASA_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins allowed to call `/api/*` with credentials. A literal `*` makes the server refuse to start, since reflected origins plus credentials would defeat the check. |

## Running backend tests

From `Nasa_Backend/`, using the project's Python environment:

```bash
python -m pytest -m "not local"           # tier 1: CPU-only, no weights/GPU needed (this is what CI runs)
python -m pytest -m "local and not slow"  # tier 2: GPU + weights; required before opening a PR
python -m pytest -m "local and slow"      # full-mode golden (~10 min); run before merging numeric changes
```

Tier 1 exercises the pure-Python/numpy modules plus the API/SSE contract against a fake model (no weights or GPU). Tier 2 loads the real weights and runs the two fast basic-mode golden masters; the `slow` marker adds the full-mode golden. Golden masters live in `Nasa_Backend/tests/golden/expected/*.json` — re-record them with `python tests/golden/record_goldens.py` only when a numeric change is intended and reviewed.

## Running the frontend

1. Change to the frontend folder and install packages:

```bash
cd nasa-frontend
npm install
```

2. Configure the frontend environment so it can reach the backend server running on port 8050.

Create a `.env` file in `nasa-frontend/` (example):

```
# Example `nasa-frontend/.env` content
REACT_APP_BACKEND_API_URL=http://localhost:8050/api
```

Make sure the frontend reads `REACT_APP_BACKEND_API_URL`. If your backend is on a different host or port, update the URL accordingly.

3. Start the frontend dev server:

```bash
npm start
```

This should open the React app (usually at http://localhost:3000). The frontend will call the backend at the URL you set in `.env`.

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
git add nasa-frontend/src/components/MyComponent.js
git add Nasa_Backend/nasa_backend/api/routes.py
```

- Useful git workflow tips:
	- Use `git status` to review changed files before adding.
	- Use `git add -p` to stage hunks interactively when appropriate.
	- If you accidentally staged large files, use `git reset <file>` to unstage them before committing.

## Files/paths you should NOT push

- Any generated segmentation videos or exports (e.g., `segmentation results/`)
- Local environment folders and virtualenvs (`.venv/`, `venv/`, `env/`)

Add appropriate patterns to your `.gitignore` to keep these out of the repo (if not already present).
