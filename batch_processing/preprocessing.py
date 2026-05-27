from pathlib import Path

import pandas as pd

# Project root (energy_project/) — this file lives at batch_processing/preprocessing.py
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "owid-energy-data.csv"  # put the raw OWID file here
CLEAN_DATA_PATH = ROOT / "data" / "europe_energy_1990_2025_cleaned.csv"

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)
# 1. Fix country naming (OWID inconsistencies)
df["country"] = df["country"].replace({"Czechia": "Czech Republic"})
# 2. European countries
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
# 3. Filter countries
df = df[df["country"].isin(europe_countries)]
# 4. Filter years (1990–2025)
df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]
# 5. Select required 20 columns
columns = [
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
df = df[columns]
# 6. Handle missing values
# Drop rows with too many missing values
df = df.dropna(thresh=12)
# Sort before filling
df = df.sort_values(["country", "year"])
# Forward fill within each country (important for time series)
df = df.groupby("country").apply(lambda x: x.ffill()).reset_index(drop=True)
# 7. Remove duplicates
df = df.drop_duplicates(subset=["country", "year"])
# 8. Remove weak countries
# Keeps only countries with enough data points
df = df.groupby("country").filter(lambda x: x["year"].nunique() >= 10)
# 9. Final formatting
df = df.sort_values(by=["country", "year"]).reset_index(drop=True)
# 10. Save cleaned dataset
CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_DATA_PATH, index=False)
print("Clean European dataset ready with 20 features!")
