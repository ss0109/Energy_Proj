from __future__ import annotations

import pandas as pd

EUROPE_COUNTRIES = [
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

COLUMNS = [
    "country",
    "year",
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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country"] = df["country"].replace({"Czechia": "Czech Republic"})
    df = df[df["country"].isin(EUROPE_COUNTRIES)]
    df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]
    df = df[COLUMNS]
    df = df.dropna(thresh=12)
    df = df.sort_values(["country", "year"])
    grouped = df.groupby("country", group_keys=False)
    filled = grouped[df.columns.drop("country")].apply(lambda x: x.ffill())
    df = pd.concat([df[["country"]], filled], axis=1)
    df = df.drop_duplicates(subset=["country", "year"])
    df = df.groupby("country").filter(lambda x: x["year"].nunique() >= 10)
    return df.sort_values(by=["country", "year"]).reset_index(drop=True)


def _make_row(country: str, year: int, **overrides):
    base = {col: 1.0 for col in COLUMNS}
    base["country"] = country
    base["year"] = year
    base.update(overrides)
    return base


def test_filters_to_european_countries():
    df = pd.DataFrame(
        [_make_row("Germany", y) for y in range(2000, 2015)]
        + [_make_row("United States", y) for y in range(2000, 2015)]
        + [_make_row("China", y) for y in range(2000, 2015)]
    )
    out = preprocess(df)
    assert set(out["country"].unique()) == {"Germany"}


def test_renames_czechia_to_czech_republic():
    df = pd.DataFrame([_make_row("Czechia", y) for y in range(2000, 2015)])
    out = preprocess(df)
    assert "Czech Republic" in out["country"].unique()
    assert "Czechia" not in out["country"].unique()


def test_year_range_filter():
    df = pd.DataFrame([_make_row("Germany", y) for y in range(1980, 2030)])
    out = preprocess(df)
    assert out["year"].min() >= 1990
    assert out["year"].max() <= 2025


def test_drops_countries_with_fewer_than_ten_years():
    df = pd.DataFrame(
        [_make_row("Germany", y) for y in range(2000, 2015)]
        + [_make_row("Malta", y) for y in range(2000, 2005)]  # only 5 years
    )
    out = preprocess(df)
    assert "Germany" in out["country"].unique()
    assert "Malta" not in out["country"].unique()


def test_drops_duplicates():
    rows = [_make_row("Germany", y) for y in range(2000, 2015)]
    rows.append(_make_row("Germany", 2005, gdp=999.0))
    df = pd.DataFrame(rows)
    out = preprocess(df)
    assert (out["country"] == "Germany").sum() == 15


def test_output_has_exactly_twenty_columns():
    df = pd.DataFrame([_make_row("Germany", y) for y in range(2000, 2015)])
    out = preprocess(df)
    assert list(out.columns) == COLUMNS
