# Contributing

This is a research-lab repository; these rules keep changes reviewable
and keep large artifacts out of git.

## Workflow

- Work on a feature branch off `main` — never commit to `main` directly.
- Open a pull request and get peer review before merging. The PR template
  lists the required verification; paste actual test summary lines, don't
  just tick boxes.
- Stage files explicitly by name. Never use `git add .` — large generated
  artifacts sit alongside source in this repo.
- Useful habits: `git status` before adding; `git add -p` to stage hunks
  interactively; if you accidentally staged something large,
  `git reset <file>` before committing.

## Testing requirements

Run from `backend/`, using the project's Python environment:

| Command | When |
|---|---|
| `python -m pytest -m "not local"` | Always (tier 1 — CI runs this on every PR) |
| `python -m pytest -m "local and not slow"` | **Required before opening any PR** (tier 2: GPU + weights) |
| `python -m pytest -m "local and slow"` | Before merging changes that touch numeric code paths (~10-min full-mode golden) |
| `ruff check .` | Always |

Run the tier-2/golden suites on an **idle GPU** — concurrent training or other
GPU jobs on the same machine make inference nondeterministic, so golden
comparisons can flake even when the code is correct.

Golden masters (`backend/tests/golden/expected/*.json`) pin numeric behavior.
Never re-record them to make a test pass; re-record them (`python
tests/golden/record_goldens.py`) only when a numeric change is intended and
has been reviewed.

## Never commit

- Model weights (`*.pt`) beyond the ones already tracked, and input/output
  videos (`*.mp4`, `*.avi`) or datasets
- Generated results: Excel/CSV summaries, `segmentation results/`,
  `backend/output/`, `<video>_charts/`, `<video>_per_frame_xlsx/`
- Local environments and editor config (`.venv/`, `venv/`, `.idea/`), and
  `frontend/.env`

The root and `backend/` `.gitignore` files cover most of these, but not every
generated artifact — check `git status` before staging.
