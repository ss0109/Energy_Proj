# Slimmer base than `python:3.10` (full Debian). ~40MB vs ~350MB at the bottom.
FROM python:3.10-slim AS base

# System deps: psycopg2-binary still wants libpq at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libpq5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps FIRST so this layer caches independently of code changes.
# Without splitting requirements from source, every code edit invalidates the
# pip layer and rebuilds take minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the application code.
COPY batch_processing/ ./batch_processing/
COPY online_deploy/ ./online_deploy/
COPY data/ ./data/
COPY db/ ./db/

# Run as non-root for safety.
RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app
USER app

EXPOSE 8000

# Basic health check Docker can read.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail --silent http://localhost:8000/ || exit 1

# Correct module path: the app is at online_deploy/app.py, not app.py.
CMD ["uvicorn", "online_deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
