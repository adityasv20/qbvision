import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

IN_PATH = "data/qb_season_dataset.csv"
OUT_PATH = "data/qbvision_ratings.csv"

def zscore_by_season(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("season")[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))

def main():
    df = pd.read_csv(IN_PATH)

    # basic safety + types
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season", "qb_name", "team"])
    df["season"] = df["season"].astype(int)

    # filter small samples
    df = df[df["dropbacks"] >= 150].copy()

    # Make sure required columns exist (or create safe fallbacks)
    needed = [
        "epa_per_db", "int_rate", "sack_rate", "ay_per_att", "yac_per_comp",
        "team_sack_rate_allowed", "team_yac_per_comp", "team_no_huddle_rate"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # Adjusted skill residuals based on context
    # Use context columns that exist in the file
    context_features = [
        "team_sack_rate_allowed",
        "team_yac_per_comp",
        "team_no_huddle_rate",
    ]
    context_features = [c for c in context_features if c in df.columns]

    Xc = df[context_features].copy()
    y = df["epa_per_db"].copy()

    context_model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    context_model.fit(Xc, y)

    df["context_pred_epa_per_db"] = context_model.predict(Xc)
    df["skill_residual"] = df["epa_per_db"] - df["context_pred_epa_per_db"]

    # proxies in case fields were missing
    df["rating_proxy"] = (
        df["epa_per_db"].fillna(0)
        - 1.5 * df["int_rate"].fillna(0)
        - 0.8 * df["sack_rate"].fillna(0)
    )

    # explosiveness proxy
    df["explosive_proxy"] = df["ay_per_att"].fillna(df["ay_per_att"].median()) + 0.15 * df["yac_per_comp"].fillna(0)

    # success proxy 
    df["success_proxy"] = df["epa_per_db"].clip(-0.4, 0.4)

    # mobility placeholder 
    df["mobility"] = df["qb_rush_epa_per_rush"] if "qb_rush_epa_per_rush" in df.columns else 0.0

    # Building subscores (season-relative z-scores)
    df["z_skill"] = zscore_by_season(df, "skill_residual")

    # efficiency bundle
    df["z_eff1"] = zscore_by_season(df, "epa_per_db")
    df["z_eff2"] = zscore_by_season(df, "success_proxy")
    df["z_eff3"] = zscore_by_season(df, "rating_proxy")
    df["efficiency_score"] = 0.5*df["z_eff1"] + 0.2*df["z_eff2"] + 0.3*df["z_eff3"]

    # decision bundle (lower means better)
    df["z_int"] = zscore_by_season(df, "int_rate")
    df["z_sack"] = zscore_by_season(df, "sack_rate")
    df["decision_score"] = -(0.6*df["z_int"] + 0.4*df["z_sack"])

    # explosiveness
    df["z_explosive"] = zscore_by_season(df, "explosive_proxy")
    df["explosiveness_score"] = df["z_explosive"]

    # mobility
    if "qb_rush_epa_per_rush" in df.columns:
        df["z_mobility"] = zscore_by_season(df, "mobility")
    else:
        df["z_mobility"] = 0.0
    df["mobility_score"] = df["z_mobility"]

    # Final QBVision Rating (season relative 0–100) 
    df["qbvision_raw"] = (
        0.35*df["z_skill"] +
        0.30*df["efficiency_score"] +
        0.20*df["decision_score"] +
        0.10*df["explosiveness_score"] +
        0.05*df["mobility_score"]
    )

    df["qbvision_z"] = zscore_by_season(df, "qbvision_raw")
    df["QBVision_Rating"] = np.clip(50 + 15*df["qbvision_z"], 0, 100)

    # Next-season target (for training)
    df = df.sort_values(["qb_name", "season"]).copy()
    df["next_QBVision_Rating"] = df.groupby("qb_name")["QBVision_Rating"].shift(-1)

    # Save output
    out_cols = [
        "season","qb_name","team","dropbacks",
        "QBVision_Rating","next_QBVision_Rating",
        "epa_per_db","context_pred_epa_per_db","skill_residual",
        "int_rate","sack_rate","ay_per_att","yac_per_comp",
        "team_sack_rate_allowed","team_yac_per_comp","team_no_huddle_rate",
        "efficiency_score","decision_score","explosiveness_score","mobility_score",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    df[out_cols].to_csv(OUT_PATH, index=False)
    print(f"Saved QBVision ratings -> {OUT_PATH} rows={len(df)}")

if __name__ == "__main__":
    main()
