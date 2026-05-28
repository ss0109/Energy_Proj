from __future__ import annotations

import pandas as pd
import pytest


def build_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.sort_values(by=["country", "year"]).copy()
    for col in features:
        df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("country")[col].shift(2)
        df[f"{col}_trend"] = df[f"{col}_lag1"] - df[f"{col}_lag2"]
    df["year_scaled"] = df["year"] - df["year"].min()
    return df.dropna()


def test_lag_features_shift_within_country(synthetic_energy_df, numeric_features):
    out = build_features(synthetic_energy_df, numeric_features)
    for country in out["country"].unique():
        country_rows = out[out["country"] == country].sort_values("year")
        first = country_rows.iloc[0]
        original = synthetic_energy_df[
            (synthetic_energy_df["country"] == country)
            & (synthetic_energy_df["year"] == first["year"] - 1)
        ]["population"].iloc[0]
        assert first["population_lag1"] == pytest.approx(original)


def test_trend_is_lag1_minus_lag2(synthetic_energy_df, numeric_features):
    """trend = lag1 - lag2 — this is the invariant the model relies on."""
    out = build_features(synthetic_energy_df, numeric_features)
    for col in numeric_features:
        diff = out[f"{col}_lag1"] - out[f"{col}_lag2"]
        pd.testing.assert_series_equal(out[f"{col}_trend"], diff, check_names=False)


def test_year_scaled_starts_at_zero(synthetic_energy_df, numeric_features):
    out = build_features(synthetic_energy_df, numeric_features)
    assert out["year_scaled"].min() >= 0
    min_year = synthetic_energy_df["year"].min()
    assert (out["year_scaled"] == out["year"] - min_year).all()


def test_dropna_removes_first_two_years_per_country(synthetic_energy_df, numeric_features):
    out = build_features(synthetic_energy_df, numeric_features)
    n_countries = synthetic_energy_df["country"].nunique()
    n_years = synthetic_energy_df["year"].nunique()
    assert len(out) == n_countries * (n_years - 2)


def test_no_cross_country_leakage(numeric_features):

    df = pd.DataFrame(
        [
            {"country": "switzerland", "year": 2020, **{f: 1.0 for f in numeric_features}},
            {"country": "switzerland", "year": 2021, **{f: 2.0 for f in numeric_features}},
            {"country": "switzerland", "year": 2022, **{f: 3.0 for f in numeric_features}},
            {"country": "germany", "year": 2020, **{f: 100.0 for f in numeric_features}},
            {"country": "germany", "year": 2021, **{f: 200.0 for f in numeric_features}},
            {"country": "germany", "year": 2022, **{f: 300.0 for f in numeric_features}},
        ]
    )
    out = build_features(df, numeric_features)

    for country, expected_lag1 in [("switzerland", 2.0), ("germany", 200.0)]:
        row = out[out["country"] == country].iloc[0]
        assert row["population_lag1"] == pytest.approx(
            expected_lag1
        ), f"{country} leaked across country boundary"
