"""FastAPI integration tests.

These tests assume two small patches to online_deploy/app.py to make it
testable:

1. The psycopg2 connection is opened LAZILY (inside a helper, not at module
   import), and is skipped entirely when DISABLE_DB_LOGGING=1.
2. The MLflow `model = mlflow.sklearn.load_model(...)` line is wrapped so
   that SKIP_MODEL_LOAD=1 (set by conftest.py) bypasses it.

The patch lives in the README — until you apply it, mark this file's tests
as skipped and the rest of the CI still runs.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

# Skip the whole file unless the app has been patched to honour the env vars.
# This way CI doesn't break before you've applied the small refactor.
pytestmark = pytest.mark.skipif(
    os.getenv("APP_PATCHED") != "1",
    reason="Set APP_PATCHED=1 once online_deploy/app.py honours DISABLE_DB_LOGGING and SKIP_MODEL_LOAD.",
)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Spin up the FastAPI app with a tiny fixture CSV and no real model/DB."""
    # Point the app at a tiny CSV so it has SOMETHING to read at startup.
    import pandas as pd

    csv_path = tmp_path / "europe_energy_1990_2025_cleaned.csv"
    rows = []
    for country in ["switzerland", "germany"]:
        for year in range(2020, 2026):
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "population": 8e6,
                    "gdp": 7e11,
                    "primary_energy_consumption": 1000.0,
                    "energy_per_capita": 35.0,
                    "energy_per_gdp": 1.4,
                    "electricity_demand": 60.0,
                    "electricity_demand_per_capita": 7.0,
                    "fossil_fuel_consumption": 600.0,
                    "fossil_electricity": 20.0,
                    "fossil_share_energy": 60.0,
                    "renewables_consumption": 200.0,
                    "renewables_electricity": 30.0,
                    "renewables_share_energy": 20.0,
                    "low_carbon_electricity": 40.0,
                    "low_carbon_share_energy": 40.0,
                    "solar_electricity": 5.0,
                    "wind_electricity": 10.0,
                    "hydro_electricity": 15.0,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    monkeypatch.setenv("DATA_PATH", str(csv_path))

    # Force re-import so the env vars are picked up.
    sys.modules.pop("online_deploy.app", None)
    from online_deploy import app as app_module

    importlib.reload(app_module)

    # Replace the model with a deterministic stub.
    class StubModel:
        def predict(self, X):
            # Return zeros of the right shape (n_rows × n_features).
            return np.zeros((len(X), 18))

    app_module.model = StubModel()

    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_rejects_past_year(client):
    response = client.post("/predict", params={"country": "switzerland", "year": 2020})
    assert response.status_code == 200
    body = response.json()
    assert "error" in body
    assert "year" in body["error"].lower()


def test_predict_rejects_unknown_country(client):
    response = client.post("/predict", params={"country": "atlantis", "year": 2030})
    assert response.status_code == 200
    assert "error" in response.json()


def test_predict_returns_all_features(client, numeric_features):
    response = client.post("/predict", params={"country": "switzerland", "year": 2030})
    assert response.status_code == 200
    body = response.json()
    if "error" in body:
        pytest.fail(f"unexpected error: {body['error']}")
    assert "prediction" in body
    # All 18 numeric features must be present in the response.
    for feature in numeric_features:
        assert feature in body["prediction"], f"missing feature: {feature}"
