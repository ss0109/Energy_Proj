"""FastAPI inference service with Postgres prediction logging.

Key changes vs the original (to make this CI/testable):
  * DB connection is lazy and skipped entirely when DISABLE_DB_LOGGING=1.
  * Model load is skipped when SKIP_MODEL_LOAD=1 (used by smoke tests).
  * Data path and Postgres credentials come from env vars with sensible
    defaults — no more hardcoded Windows paths.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# ---------------------------------------------------------------------------
# Configuration (env-driven, with safe defaults for local dev).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "europe_energy_1990_2025_cleaned.csv"))
MLRUNS_URI = os.getenv("MLRUNS_URI", f"file:{ROOT / 'mlruns'}")
RUN_ID = os.getenv("MLFLOW_RUN_ID", "7c855f5fcf5a4a6193976fa7a5a4aa89")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "energy_monitoring")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "postgres")

DISABLE_DB_LOGGING = os.getenv("DISABLE_DB_LOGGING", "0") == "1"
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"


# ---------------------------------------------------------------------------
# Model loading (skippable for CI smoke tests).
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLRUNS_URI)

if SKIP_MODEL_LOAD:
    model = None  # type: ignore[assignment]
else:
    model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")


# ---------------------------------------------------------------------------
# Lazy Postgres connection — opened on first /predict, not at import time.
# ---------------------------------------------------------------------------
_conn = None


def get_conn():
    """Return a singleton psycopg2 connection, or None when DB logging is disabled."""
    global _conn
    if DISABLE_DB_LOGGING:
        return None
    if _conn is None or _conn.closed:
        import psycopg2  # imported lazily so tests don't need psycopg2 installed

        _conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS,
        )
    return _conn


# ---------------------------------------------------------------------------
# Data + feature column setup.
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["country"] = df["country"].str.strip().str.lower()
country_col = "country"
year_col = "year"
df = df.sort_values(by=[country_col, year_col])
features = df.drop(columns=[country_col, year_col]).columns

feature_cols = (
    [country_col, "year_scaled"]
    + [f"{col}_lag1" for col in features]
    + [f"{col}_lag2" for col in features]
    + [f"{col}_trend" for col in features]
)


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "API Running",
        "model_loaded": model is not None,
        "db_logging": not DISABLE_DB_LOGGING,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(country: str, year: int):
    try:
        start_time = time.time()
        country = country.lower()

        country_data = df[df[country_col] == country].sort_values(by=year_col)
        if len(country_data) < 2:
            return {"error": "Not enough data"}

        current_year = int(country_data.iloc[-1][year_col])
        if year <= current_year:
            return {"error": f"Enter year > {current_year}"}

        history = country_data.tail(2).copy()

        # Recursive year-by-year forecasting.
        for future_year in range(current_year + 1, year + 1):
            last = history.iloc[-1]
            second_last = history.iloc[-2]
            year_scaled = future_year - df[year_col].min()

            input_data = {"country": country, "year_scaled": year_scaled}
            for col in features:
                lag1 = 0 if pd.isna(last[col]) else last[col]
                lag2 = 0 if pd.isna(second_last[col]) else second_last[col]
                input_data[f"{col}_lag1"] = lag1
                input_data[f"{col}_lag2"] = lag2
                input_data[f"{col}_trend"] = lag1 - lag2

            input_df = pd.DataFrame([input_data])[feature_cols]
            prediction = model.predict(input_df)[0]
            prediction = np.nan_to_num(prediction)

            new_row = {country_col: country, year_col: future_year}
            for i, col in enumerate(features):
                new_row[col] = prediction[i]
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        final_prediction = {k: float(v) for k, v in history.iloc[-1][features].to_dict().items()}
        latency = time.time() - start_time

        # Log to Postgres unless disabled.
        conn = get_conn()
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_logs (
                        country, year,
                        population, gdp, primary_energy_consumption,
                        energy_per_capita, energy_per_gdp,
                        electricity_demand, electricity_demand_per_capita,
                        fossil_fuel_consumption, fossil_electricity, fossil_share_energy,
                        renewables_consumption, renewables_electricity, renewables_share_energy,
                        low_carbon_electricity, low_carbon_share_energy,
                        solar_electricity, wind_electricity, hydro_electricity,
                        latency_s
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        country,
                        year,
                        final_prediction.get("population"),
                        final_prediction.get("gdp"),
                        final_prediction.get("primary_energy_consumption"),
                        final_prediction.get("energy_per_capita"),
                        final_prediction.get("energy_per_gdp"),
                        final_prediction.get("electricity_demand"),
                        final_prediction.get("electricity_demand_per_capita"),
                        final_prediction.get("fossil_fuel_consumption"),
                        final_prediction.get("fossil_electricity"),
                        final_prediction.get("fossil_share_energy"),
                        final_prediction.get("renewables_consumption"),
                        final_prediction.get("renewables_electricity"),
                        final_prediction.get("renewables_share_energy"),
                        final_prediction.get("low_carbon_electricity"),
                        final_prediction.get("low_carbon_share_energy"),
                        final_prediction.get("solar_electricity"),
                        final_prediction.get("wind_electricity"),
                        final_prediction.get("hydro_electricity"),
                        latency,
                    ),
                )
                conn.commit()

        return {"country": country, "year": year, "prediction": final_prediction}

    except Exception as e:
        conn = get_conn()
        if conn is not None:
            conn.rollback()
        return {"error": str(e)}
