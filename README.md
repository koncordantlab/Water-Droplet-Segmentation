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

(Override the host/port with the `NASA_HOST` / `NASA_PORT` environment variables. Other optional variables — `NASA_WEIGHTS_PATH`, `NASA_WEBUI_DIR`, `NASA_VIDEO_ROOTS`, `NASA_CORS_ORIGINS` — are documented in `CLAUDE.md`.)

## Running backend tests

From `Nasa_Backend/`, using the project's Python environment:

```bash
python -m pytest -m "not local"           # tier 1: CPU-only, no weights/GPU needed (this is what CI runs)
python -m pytest -m "local and not slow"  # tier 2: GPU + weights; required before opening a PR
python -m pytest -m "local and slow"      # full-mode golden (~10 min); run before merging numeric changes
```

See `CLAUDE.md` for what each tier covers and where the golden masters live.

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
