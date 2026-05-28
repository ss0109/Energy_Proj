"""Shared pytest fixtures.

Kept tiny on purpose — the goal is fast, deterministic unit tests that don't
need the real OWID dataset or a trained model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _disable_db_logging(monkeypatch):
    """Make sure no test accidentally opens a real Postgres connection at import."""
    monkeypatch.setenv("DISABLE_DB_LOGGING", "1")
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")


@pytest.fixture
def synthetic_energy_df() -> pd.DataFrame:
    """A small but structurally correct stand-in for europe_energy_1990_2025_cleaned.csv.

    Two countries, ten years each, all required columns. Values are linear in
    `year` per country so tests can make exact assertions about lag/trend
    behaviour.
    """
    countries = ["switzerland", "germany"]
    years = list(range(2014, 2024))
    rows = []
    for c_idx, country in enumerate(countries):
        for y_idx, year in enumerate(years):
            # Distinct value per (country, year) makes drift/lag bugs obvious.
            base = 100.0 + 10 * c_idx + y_idx
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "population": base * 1e6,
                    "gdp": base * 1e9,
                    "primary_energy_consumption": base * 10,
                    "energy_per_capita": base,
                    "energy_per_gdp": base / 100,
                    "electricity_demand": base * 5,
                    "electricity_demand_per_capita": base / 2,
                    "fossil_fuel_consumption": base * 7,
                    "fossil_electricity": base * 3,
                    "fossil_share_energy": 60 - y_idx,  # declining
                    "renewables_consumption": base * 2,
                    "renewables_electricity": base * 1.5,
                    "renewables_share_energy": 20 + y_idx,  # rising
                    "low_carbon_electricity": base * 4,
                    "low_carbon_share_energy": 30 + y_idx,
                    "solar_electricity": base * 0.5,
                    "wind_electricity": base * 1.0,
                    "hydro_electricity": base * 2.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def numeric_features() -> list[str]:
    """The 18 numeric feature columns (everything except country/year)."""
    return [
        "population",
        "gdp",
        "primary_energy_consumption",
        "energy_per_capita",
        "energy_per_gdp",
        "electricity_demand",
        "electricity_demand_per_capita",
        "fossil_fuel_consumption",
        "fossil_electricity",
        "fossil_share_energy",
        "renewables_consumption",
        "renewables_electricity",
        "renewables_share_energy",
        "low_carbon_electricity",
        "low_carbon_share_energy",
        "solar_electricity",
        "wind_electricity",
        "hydro_electricity",
    ]


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG so tests are reproducible."""
    return np.random.default_rng(42)
