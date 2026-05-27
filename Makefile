.PHONY: help build up down logs train pipeline grafana adminer api shell clean \
        lint format test test-cov install-dev ci

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Docker / stack:"
	@echo "  make up         - build images and start postgres/adminer/grafana/api"
	@echo "  make pipeline   - run the Prefect flow (batch predict -> drift -> backtest)"
	@echo "  make logs       - tail logs of all running services"
	@echo "  make down       - stop services (keeps volumes)"
	@echo "  make clean      - stop services AND wipe volumes"
	@echo ""
	@echo "Dev / CI:"
	@echo "  make install-dev  - install runtime + dev dependencies"
	@echo "  make lint         - ruff check + ruff format check"
	@echo "  make format       - ruff format (auto-fix)"
	@echo "  make test         - run pytest"
	@echo "  make test-cov     - run pytest with coverage report"
	@echo "  make ci           - everything CI runs locally (lint + test)"

# ---------------------------------------------------------------------------
# Docker / runtime
# ---------------------------------------------------------------------------
build:
	docker compose build

up:
	docker compose up -d --build
	@echo ""
	@echo "Services starting. First-time training takes ~2-3 minutes."
	@echo "Run 'make logs' to follow progress."

pipeline:
	docker compose --profile manual run --rm pipeline

logs:
	docker compose logs -f --tail=200

grafana:
	@echo "Open http://localhost:3000"

adminer:
	@echo "Open http://localhost:8080  System=PostgreSQL Server=postgres User=postgres Pass=postgres DB=energy_monitoring"

api:
	@echo "Open http://localhost:8000/docs"

shell:
	docker compose run --rm api shell

down:
	docker compose --profile manual down

clean:
	docker compose --profile manual down -v

# ---------------------------------------------------------------------------
# Local dev / CI mirror
# ---------------------------------------------------------------------------
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -v

test-cov:
	pytest -v --cov=. --cov-report=term-missing --cov-report=html
	@echo "HTML coverage report at htmlcov/index.html"

# Mirrors what GitHub Actions runs — use this before pushing.
ci: lint test
	@echo ""
	@echo "✓ Local CI passed. Safe to push."
