import pandas as pd
import numpy as np
import nfl_data_py as nfl

SEASONS = list(range(2018, 2026))
MIN_QB_DROPBACKS = 150
OUT_PATH = "data/qb_season_dataset.csv"

def main():
    pbp = nfl.import_pbp_data(SEASONS, downcast=True)

    # Regular season only
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    # Identify the qb
    pbp["qb_name"] = pbp["passer_player_name"].fillna(pbp["rusher_player_name"])
    pbp = pbp[pbp["qb_name"].notna()].copy()

    # Sorting through dropbacks
    if "qb_dropback" in pbp.columns:
        pbp["dropback"] = pbp["qb_dropback"].fillna(0).astype(int)
    else:
        pbp["dropback"] = (
            pbp["pass_attempt"].fillna(0).astype(int) +
            pbp["sack"].fillna(0).astype(int)
        )

    # Dropbacks only
    pbp_db = pbp[pbp["dropback"] == 1].copy()

    # Flags
    pbp_db["is_sack"] = pbp_db["sack"].fillna(0).astype(int) if "sack" in pbp_db.columns else 0
    pbp_db["is_int"] = pbp_db["interception"].fillna(0).astype(int) if "interception" in pbp_db.columns else 0
    pbp_db["is_pass_attempt"] = pbp_db["pass_attempt"].fillna(0).astype(int) if "pass_attempt" in pbp_db.columns else 0

    # No-huddle-rate (ability to work under pressure)
    pbp_db["is_no_huddle"] = pbp_db["no_huddle"].fillna(0).astype(int) if "no_huddle" in pbp_db.columns else 0

    # Team on offense
    pbp_db["off_team"] = pbp_db["posteam"]

    # QB season totals
    qb_season = (
        pbp_db.groupby(["season", "qb_name"], as_index=False)
        .agg(
            dropbacks=("dropback", "sum"),
            epa=("epa", "sum"),
            epa_per_db=("epa", "mean"),
            ints=("is_int", "sum"),
            sacks=("is_sack", "sum"),
            air_yards=("air_yards", "sum"),
            comp=("complete_pass", "sum"),
            att=("is_pass_attempt", "sum"),
            yac=("yards_after_catch", "sum"),
        )
    )

    qb_season["int_rate"] = qb_season["ints"] / qb_season["dropbacks"].replace(0, np.nan)
    qb_season["sack_rate"] = qb_season["sacks"] / qb_season["dropbacks"].replace(0, np.nan)
    qb_season["ay_per_att"] = qb_season["air_yards"] / qb_season["att"].replace(0, np.nan)
    qb_season["yac_per_comp"] = qb_season["yac"] / qb_season["comp"].replace(0, np.nan)

    # Protection Rates
    team_protection = (
        pbp_db.groupby(["season", "off_team"], as_index=False)
        .agg(team_dropbacks=("dropback", "sum"), team_sacks=("is_sack", "sum"))
    )
    team_protection["team_sack_rate_allowed"] = (
        team_protection["team_sacks"] / team_protection["team_dropbacks"].replace(0, np.nan)
    )

    # Weapons explosiveness
    team_weapons = (
        pbp_db.groupby(["season", "off_team"], as_index=False)
        .agg(team_yac=("yards_after_catch", "sum"), team_completions=("complete_pass", "sum"))
    )
    team_weapons["team_yac_per_comp"] = team_weapons["team_yac"] / team_weapons["team_completions"].replace(0, np.nan)

    # Under pressure (no-huddle-rate)
    team_scheme = (
        pbp_db.groupby(["season", "off_team"], as_index=False)
        .agg(db=("dropback", "sum"), no_huddle_plays=("is_no_huddle", "sum"))
    )
    team_scheme["team_no_huddle_rate"] = team_scheme["no_huddle_plays"] / team_scheme["db"].replace(0, np.nan)

    # Assigning QB to their primary team
    qb_team = (
        pbp_db.groupby(["season", "qb_name", "off_team"], as_index=False)
        .agg(db=("dropback", "sum"))
        .sort_values(["season", "qb_name", "db"], ascending=[True, True, False])
        .drop_duplicates(["season", "qb_name"])
        .rename(columns={"off_team": "team"})
    )

    df = qb_season.merge(qb_team, on=["season", "qb_name"], how="left")

    df = df.merge(team_protection.rename(columns={"off_team": "team"}), on=["season", "team"], how="left")
    df = df.merge(team_weapons.rename(columns={"off_team": "team"}), on=["season", "team"], how="left")
    df = df.merge(team_scheme.rename(columns={"off_team": "team"}), on=["season", "team"], how="left")

    # Filter samples
    df = df[df["dropbacks"] >= MIN_QB_DROPBACKS].copy()

    # Next-season target 
    df = df.sort_values(["qb_name", "season"]).copy()
    df["next_epa_per_db"] = df.groupby("qb_name")["epa_per_db"].shift(-1)

    # Experience by seasons
    df["seasons_seen"] = df.groupby("qb_name").cumcount() + 1

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved dataset: {OUT_PATH} rows={len(df)} seasons={df['season'].min()}-{df['season'].max()}")

if __name__ == "__main__":
    main()
