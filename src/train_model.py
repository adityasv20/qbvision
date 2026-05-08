import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/qbvision_ratings.csv"
MODEL_PATH = "data/qb_model.joblib"

FEATURES = [
    "QBVision_Rating",
    "epa_per_db",
    "skill_residual",
    "int_rate",
    "sack_rate",
    "ay_per_att",
    "yac_per_comp",
    "team_sack_rate_allowed",
    "team_yac_per_comp",
    "team_no_huddle_rate",
    "dropbacks",
]

TARGET = "next_QBVision_Rating"

# Blend predictions with current rating to reduce unrealistic mean-reversion
PRED_BLEND_MODEL = 0.65
PRED_BLEND_CURRENT = 0.35

def main():
    df = pd.read_csv(DATA_PATH)

    # Ensure numbers where they are needed
    for c in FEATURES + [TARGET, "season"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Only rows where we have next season label
    df = df.dropna(subset=[TARGET]).copy()

    df = df.sort_values(["season"]).copy()

    # Drop any feature columns missing from the file
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in {DATA_PATH}: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            random_state=42,
            n_estimators=350,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9
        ))
    ])

    # Time-based validation (good practice)
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Blend to stabilize projections
        preds = (PRED_BLEND_MODEL * preds) + (PRED_BLEND_CURRENT * X_test["QBVision_Rating"].values)

        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)

    print("MAE by split:", [round(m, 4) for m in maes])
    print("Avg MAE:", round(float(np.mean(maes)), 4))
    print(f"Blend: {PRED_BLEND_MODEL:.2f}*model + {PRED_BLEND_CURRENT:.2f}*current_rating")

    # Fit on all data + save
    model.fit(X, y)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "pred_blend_model": PRED_BLEND_MODEL,
            "pred_blend_current": PRED_BLEND_CURRENT,
        },
        MODEL_PATH
    )
    print(f"Saved model bundle -> {MODEL_PATH}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
