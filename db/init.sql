-- Postgres initialisation. This file is mounted into
-- /docker-entrypoint-initdb.d/ inside the official postgres image, so it runs
-- automatically the first time the database starts (when the volume is empty).

CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    country TEXT NOT NULL,
    year INT NOT NULL,
    population DOUBLE PRECISION,
    gdp DOUBLE PRECISION,
    primary_energy_consumption DOUBLE PRECISION,
    energy_per_capita DOUBLE PRECISION,
    energy_per_gdp DOUBLE PRECISION,
    electricity_demand DOUBLE PRECISION,
    electricity_demand_per_capita DOUBLE PRECISION,
    fossil_fuel_consumption DOUBLE PRECISION,
    fossil_electricity DOUBLE PRECISION,
    fossil_share_energy DOUBLE PRECISION,
    renewables_consumption DOUBLE PRECISION,
    renewables_electricity DOUBLE PRECISION,
    renewables_share_energy DOUBLE PRECISION,
    low_carbon_electricity DOUBLE PRECISION,
    low_carbon_share_energy DOUBLE PRECISION,
    solar_electricity DOUBLE PRECISION,
    wind_electricity DOUBLE PRECISION,
    hydro_electricity DOUBLE PRECISION,
    latency_s DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_country  ON prediction_logs (country);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_year     ON prediction_logs (year);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created  ON prediction_logs (created_at);


CREATE TABLE IF NOT EXISTS drift_logs (
    id SERIAL PRIMARY KEY,
    feature TEXT NOT NULL,
    ks_stat DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    psi DOUBLE PRECISION,
    drift_detected BOOLEAN,
    ref_mean DOUBLE PRECISION,
    cur_mean DOUBLE PRECISION,
    n_reference INT,
    n_current INT,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_logs_feature ON drift_logs (feature);
CREATE INDEX IF NOT EXISTS idx_drift_logs_created ON drift_logs (created_at);