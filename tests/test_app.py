from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_PATCHED") != "1",
    reason="Set APP_PATCHED=1 once online_deploy/app.py honours DISABLE_DB_LOGGING and SKIP_MODEL_LOAD.",
)


@pytest.fixture
def client(monkeypatch, tmp_path):
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

    sys.modules.pop("online_deploy.app", None)
    from online_deploy import app as app_module

    importlib.reload(app_module)

    class StubModel:
        def predict(self, X):
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
    for feature in numeric_features:
        assert feature in body["prediction"], f"missing feature: {feature}"
