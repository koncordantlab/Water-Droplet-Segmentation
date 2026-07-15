## Summary

<!-- What does this PR do, and why? One short paragraph. -->

## Changes

<!-- Bullet the notable changes. Call out anything breaking: env vars, API payloads, file formats, CLI flags. -->

-

## Verification

<!-- Run from backend/ in the project Python env. Paste the actual summary lines, don't just tick. -->

- [ ] Tier 1 green: `python -m pytest -m "not local"` (CI runs this on the PR too)
- [ ] Tier 2 green on an **idle GPU**: `python -m pytest -m "local and not slow"` — required before opening any PR
- [ ] Lint clean: `ruff check .`
- [ ] Slow full-mode golden green: `python -m pytest -m "local and slow"` — **only if numeric code paths changed**
- [ ] Frontend builds: `npm run build` — only if `frontend/` changed

## Checklist

- [ ] No weights (`*.pt`), videos, or generated artifacts staged — files staged explicitly by name, never `git add .`
- [ ] Golden masters (`backend/tests/golden/expected/*.json`) untouched — or the re-record is intentional, reviewed, and explained above
- [ ] README updated if commands, env vars, or user-visible behavior changed
