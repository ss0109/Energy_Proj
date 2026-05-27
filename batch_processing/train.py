from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project root (energy_project/) — this file lives at batch_processing/train.py
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "europe_energy_1990_2025_cleaned.csv"
MLRUNS_URI = f"file:{ROOT / 'mlruns'}"

# MLflow Setup
mlflow.set_tracking_uri(MLRUNS_URI)
mlflow.set_experiment("Energy_Prediction_Model")
# LOAD DATA
df = pd.read_csv(DATA_PATH)
df["country"] = df["country"].str.strip().str.lower()
country_col = "country"
year_col = "year"
# FEATURE ENGINEERING
features = df.drop(columns=[country_col, year_col]).columns
df = df.sort_values(by=[country_col, year_col])
for col in features:
    df[f"{col}_lag1"] = df.groupby(country_col)[col].shift(1)
    df[f"{col}_lag2"] = df.groupby(country_col)[col].shift(2)
    df[f"{col}_trend"] = df[f"{col}_lag1"] - df[f"{col}_lag2"]
df["year_scaled"] = df[year_col] - df[year_col].min()
df = df.dropna()
feature_cols = (
    [country_col, "year_scaled"]
    + [f"{col}_lag1" for col in features]
    + [f"{col}_lag2" for col in features]
    + [f"{col}_trend" for col in features]
)
# STRICT TIME SPLIT
train = df[df[year_col] <= 2019]
test = df[df[year_col] == 2020]
X_train = train[feature_cols]
X_test = test[feature_cols]
Y_train = train[features]
Y_test = test[features]
# PREPROCESSOR
preprocessor = ColumnTransformer(
    transformers=[("country", OneHotEncoder(handle_unknown="ignore"), [country_col])],
    remainder="passthrough",
)
# MODELS
models_to_try = {
    "RandomForest": MultiOutputRegressor(
        RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42)
    ),
    "GradientBoosting": MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=150, learning_rate=0.05)
    ),
}
best_r2 = -np.inf
best_run_id = None
best_model_name = None
best_mae = None
best_nrmse = None
# EXPERIMENT LOOP
for model_name, model in models_to_try.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, Y_train)
        Y_pred = pipeline.predict(X_test)
        # METRICS
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        # Normalize RMSE (important for large values)
        nrmse = rmse / (Y_test.max().max() - Y_test.min().min())
        # LOGGING
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("NRMSE", nrmse)
        mlflow.sklearn.log_model(pipeline, "model")
        run_id = mlflow.active_run().info.run_id
        print(
            f"{model_name} | R2: {round(r2,3)} | MAE: {round(mae,2)} | NRMSE: {round(nrmse,4)} | Run ID: {run_id}"
        )
        # BEST MODEL SELECTION
        if r2 > best_r2:
            best_r2 = r2
            best_run_id = run_id
            best_model_name = model_name
            best_mae = mae
            best_nrmse = nrmse
# FINAL RESULT
print("\nBEST MODEL")
print("Model:", best_model_name)
print("R2 Score:", round(best_r2, 3))
print("MAE:", round(best_mae, 2))
print("NRMSE:", round(best_nrmse, 4))
print("Run ID:", best_run_id)
