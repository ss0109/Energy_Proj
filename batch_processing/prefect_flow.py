import subprocess
import sys
from pathlib import Path

import requests
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

# Project root (energy_project/) — this file lives at batch_processing/prefect_flow.py
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
DATA_PATH = ROOT / "data" / "europe_energy_1990_2025_cleaned.csv"
BATCHES_DIR = ROOT / "batches"


@task(log_prints=True)
def training_task():
    print("Starting model training...")
    # Use sys.executable + absolute path so this works no matter where Prefect
    # launches the flow from.
    result = subprocess.run([sys.executable, str(TRAIN_SCRIPT)], capture_output=True, text=True)
    output = result.stdout
    print(output)

    best_model = None
    r2_score = None
    run_id = None

    for line in output.split("\n"):
        if "Model:" in line:
            best_model = line.split(":")[1].strip()
        if "R2 Score:" in line:
            r2_score = line.split(":")[1].strip()
        if "Run ID:" in line:
            run_id = line.split(":")[1].strip()

    artifact_text = f"""
### Best Model Result
**Best Model:** {best_model}
**R² Score:** {r2_score}
**Run ID:** {run_id}
"""
    create_markdown_artifact(key="best-model-result", markdown=artifact_text)
    print("Training completed")


# UPDATED TASK
@task(log_prints=True)
def batch_prediction_task():
    import random
    import uuid

    import pandas as pd

    europe_countries = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Poland",
        "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "United Kingdom",
        "Switzerland",
        "Iceland",
        "Norway",
    ]

    print("Creating batch...")

    # 1. Random country
    country = random.choice(europe_countries).lower()

    # 2. Continuous 5 years
    current_year = 2025
    start_year = random.randint(current_year + 1, current_year + 3)
    years = list(range(start_year, start_year + 5))

    print(f"Selected Country: {country}")
    print(f"Years: {years}")

    # 3. Create batch
    batch = [{"country": country, "year": y} for y in years]

    # 4. Save CSV to ROOT/batches/ so files always land in the same place
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = str(uuid.uuid4())[:8]
    batch_csv_path = BATCHES_DIR / f"batch_{batch_id}.csv"
    df = pd.DataFrame(batch)
    df.to_csv(batch_csv_path, index=False)
    print(f"Batch saved: {batch_csv_path}")

    # 5. Send to FastAPI
    print("Sending batch to FastAPI...")
    for item in batch:
        response = requests.post("http://127.0.0.1:8000/predict", params=item)
        print(response.json())

    print("Batch predictions completed")


# NEW TASK — drift detection / monitoring
@task(log_prints=True)
def monitoring_task():
    """
    Compare the latest batch of predictions in `prediction_logs` against the
    SAME countries' most recent real years (2020-2024) in the historical CSV.

    Drift rule: a feature is flagged ONLY if BOTH
      (a) the KS test or PSI says the distributions differ, AND
      (b) the means actually moved by more than 25% in relative terms.
    Condition (b) suppresses false positives caused by tiny sample sizes
    (~5 vs ~5), where the KS test's minimum p-value is mechanically <0.01
    even when nothing meaningful has changed.
    """
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import psycopg2
    from scipy import stats

    # ---- config -----------------------------------------------------------
    CSV_PATH = DATA_PATH  # ROOT-anchored, defined at module top

    PG_HOST = "localhost"
    PG_PORT = 5432
    PG_DB = "energy_monitoring"
    PG_USER = "postgres"
    PG_PASS = "postgres"

    RECENT_YEARS_FROM = 2020
    RECENT_YEARS_TO = 2024

    # Minimum relative change in the mean to call something "drifted".
    MIN_RELATIVE_CHANGE = 0.25

    FEATURES = [
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

    # ---- statistics helpers ----------------------------------------------
    def ks_test(reference, current):
        reference = np.asarray(reference, dtype=float)
        current = np.asarray(current, dtype=float)
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        if len(reference) < 2 or len(current) < 2:
            return float("nan"), float("nan")
        stat, p = stats.ks_2samp(reference, current)
        return float(stat), float(p)

    def psi(reference, current, bins=10):
        reference = np.asarray(reference, dtype=float)
        current = np.asarray(current, dtype=float)
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        if len(reference) == 0 or len(current) == 0:
            return float("nan")

        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf

        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)
        ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
        cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    # ---- connect ----------------------------------------------------------
    print("[monitoring] connecting to Postgres...")
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )

    # Make sure the drift_logs table exists
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drift_logs (
                id SERIAL PRIMARY KEY,
                feature        TEXT NOT NULL,
                ks_stat        DOUBLE PRECISION,
                p_value        DOUBLE PRECISION,
                psi            DOUBLE PRECISION,
                drift_detected BOOLEAN,
                ref_mean       DOUBLE PRECISION,
                cur_mean       DOUBLE PRECISION,
                n_reference    INT,
                n_current      INT,
                source         TEXT,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.commit()

    try:
        # ---- load current data: latest batch from prediction_logs ----------
        print("[monitoring] loading latest predictions from prediction_logs...")
        query = f"""
            SELECT {", ".join(FEATURES)}, country
            FROM prediction_logs
            WHERE created_at >= (
                SELECT MAX(created_at) - INTERVAL '10 minutes'
                FROM prediction_logs
            )
        """
        current_df = pd.read_sql(query, conn)
        print(f"[monitoring] latest batch size: {len(current_df)} rows")

        if len(current_df) == 0:
            print("[monitoring] no recent predictions — skipping drift check")
            return

        # ---- which countries are in this batch? ----------------------------
        batch_countries = current_df["country"].str.lower().unique().tolist()
        print(f"[monitoring] batch countries: {batch_countries}")

        # ---- load reference: SAME countries, recent real years -----------
        print(
            f"[monitoring] loading reference data ({RECENT_YEARS_FROM}-{RECENT_YEARS_TO}) "
            f"for the same countries as the batch"
        )
        full_df = pd.read_csv(CSV_PATH)
        full_df["country"] = full_df["country"].str.strip().str.lower()
        reference_df = full_df[
            (full_df["year"] >= RECENT_YEARS_FROM)
            & (full_df["year"] <= RECENT_YEARS_TO)
            & (full_df["country"].isin(batch_countries))
        ]
        print(f"[monitoring] reference size (same countries): {len(reference_df)} rows")

        if len(reference_df) < 2:
            print("[monitoring] not enough reference data — skipping drift check")
            return

        # Drop the country column from current_df now that we have the filter
        current_df = current_df.drop(columns=["country"])

        # ---- compute drift per feature -------------------------------------
        rows = []
        for feature in FEATURES:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            ref = reference_df[feature].dropna().to_numpy()
            cur = current_df[feature].dropna().to_numpy()
            if len(ref) < 2 or len(cur) < 2:
                continue

            ks_stat, p_value = ks_test(ref, cur)
            psi_value = psi(ref, cur)

            ref_mean_val = float(np.mean(ref))
            cur_mean_val = float(np.mean(cur))

            # Smart drift rule: KS/PSI says different AND mean moved >25%
            if abs(ref_mean_val) > 1e-9:
                relative_diff = abs(cur_mean_val - ref_mean_val) / abs(ref_mean_val)
            else:
                relative_diff = 0.0

            statistical_signal = (p_value < 0.05) or (psi_value >= 0.10)
            meaningful_change = relative_diff > MIN_RELATIVE_CHANGE
            drifted = bool(statistical_signal and meaningful_change)

            rows.append(
                {
                    "feature": feature,
                    "ks_stat": round(ks_stat, 4),
                    "p_value": round(p_value, 4),
                    "psi": round(psi_value, 4),
                    "drift_detected": drifted,
                    "ref_mean": ref_mean_val,
                    "cur_mean": cur_mean_val,
                    "n_reference": int(len(ref)),
                    "n_current": int(len(cur)),
                }
            )

        report = pd.DataFrame(rows)
        if report.empty:
            print("[monitoring] no features were comparable — nothing to log")
            return

        n_drifted = int(report["drift_detected"].sum())
        print(
            f"\n=== Drift Report ({n_drifted}/{len(report)} features drifted, "
            f"min relative change {int(MIN_RELATIVE_CHANGE * 100)}%) ==="
        )
        print(
            report[
                ["feature", "psi", "p_value", "drift_detected", "ref_mean", "cur_mean"]
            ].to_string(index=False)
        )

        # ---- write drift report to Postgres --------------------------------
        # 'source' encodes which countries the report is for, useful in Adminer
        source_label = f"prefect_batch_{'_'.join(sorted(batch_countries))[:60]}"
        with conn.cursor() as cur:
            for _, r in report.iterrows():
                cur.execute(
                    """
                    INSERT INTO drift_logs
                      (feature, ks_stat, p_value, psi, drift_detected,
                       ref_mean, cur_mean, n_reference, n_current,
                       source, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        r["feature"],
                        r["ks_stat"],
                        r["p_value"],
                        r["psi"],
                        r["drift_detected"],
                        r["ref_mean"],
                        r["cur_mean"],
                        r["n_reference"],
                        r["n_current"],
                        source_label,
                        datetime.utcnow(),
                    ),
                )
            conn.commit()
        print(f"[monitoring] wrote {len(report)} rows to drift_logs")

        # ---- attach a markdown artifact to the Prefect run -----------------
        artifact_md = (
            "### Drift Report\n\n"
            f"**Countries in batch:** {', '.join(batch_countries)}\n\n"
            f"**{n_drifted} of {len(report)} features show drift** "
            f"(statistical signal AND mean change > "
            f"{int(MIN_RELATIVE_CHANGE * 100)}%)\n\n"
            "```\n"
            + report[
                ["feature", "psi", "p_value", "drift_detected", "ref_mean", "cur_mean"]
            ].to_string(index=False)
            + "\n```"
        )
        create_markdown_artifact(key="drift-report", markdown=artifact_md)

    finally:
        conn.close()
        print("[monitoring] done")


@flow(name="energy_ml_pipeline")
def energy_pipeline():
    training_task()
    batch_prediction_task()
    monitoring_task()


if __name__ == "__main__":
    energy_pipeline()
