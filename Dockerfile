FROM python:3.10-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libpq5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only — data is mounted/generated at runtime, never baked in
COPY batch_processing/ ./batch_processing/
COPY online_deploy/ ./online_deploy/
COPY db/ ./db/

EXPOSE 8000

CMD ["uvicorn", "online_deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
