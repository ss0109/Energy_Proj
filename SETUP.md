# Setup — what to do with this zip

## Step 1 — Extract over your existing local folder

Unzip on top of your `energy_project/` folder. Windows will ask whether to
overwrite `Dockerfile`, `Makefile`, `.dockerignore`, and `online_deploy/app.py`
— say **yes to all**. These are fixed versions.

Files this zip adds: `.github/`, `tests/`, `pyproject.toml`,
`requirements-dev.txt`, `.gitignore`, `SETUP.md`.

## Step 2 — Delete the duplicates you have lying around

Your local folder still has obsolete copies of the same code in other places.
Delete these so the repo has one canonical implementation:

- The `src/` folder (whole thing) — duplicate of batch_processing + online_deploy
- Root-level Python files if they exist: `app.py`, `train.py`, `prefect_flow.py`,
  `preprocessing.py`, `deploy.py`
- Root-level `europe_energy_1990_2025_cleaned.csv` — the real one is in `data/`
- Any leftover `batch_*.csv` files in the root

## Step 3 — Verify locally before pushing

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
make ci
```

If you see `✓ Local CI passed`, GitHub Actions will pass too.

## Step 4 — Push to GitHub

```bash
git add .
# Remove obsolete files from the repo if they were tracked:
git rm -rf src/ 2>/dev/null || true
git rm app.py train.py prefect_flow.py preprocessing.py deploy.py 2>/dev/null || true
git rm europe_energy_1990_2025_cleaned.csv 2>/dev/null || true
git commit -m "Add CI/CD pipeline; consolidate to batch_processing + online_deploy"
git push
```

Open the **Actions** tab on GitHub. The `CI/CD` workflow runs automatically.

---

## What's in this zip

- `.github/workflows/ci-cd.yml` — lint → tests → docker build → push to GHCR
- `.github/workflows/train.yml` — manual + weekly retrain with R² >= 0.75 gate
- `tests/` — 20 tests: drift math (KS + PSI), feature engineering, preprocessing, FastAPI
- `pyproject.toml` — ruff + pytest config
- `Dockerfile` — fixed entrypoint (`online_deploy.app:app`), layer caching, non-root, healthcheck
- `online_deploy/app.py` — patched: lazy DB connection, env-driven config
- `Makefile` — adds `make ci`, `make test`, `make lint`, `make format`

## Daily commands

```
make ci         # what GitHub runs — do this before pushing
make test       # just tests
make lint       # just lint
make format     # auto-fix formatting
make up         # start full docker stack
make pipeline   # run the Prefect flow
```

## Gotchas

1. `tests/test_app.py` is skipped by default (`APP_PATCHED=1` not set). After your first successful push, set `APP_PATCHED: "1"` in the test job env block in `ci-cd.yml` to enable those 4 tests.
2. The CD job (push image to GHCR) only runs on push to `main` and version tags — not on PRs.
3. CD pushes to `ghcr.io/<your-username>/<repo>:latest`. No credentials needed; it uses the built-in `GITHUB_TOKEN`. Make the package public in Settings → Packages if your repo is private.
4. `train.yml` runs `python -m batch_processing.train`. The training script needs `data/europe_energy_1990_2025_cleaned.csv` to be committed in the repo — it already is in this zip.
