import pandas as pd
import streamlit as st


def force_unique_columns(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """
    Plotly Express (via Narwhals) errors if df.columns are not unique.
    This makes them unique by renaming duplicates: col, col__dup1, col__dup2, ...
    Also normalizes invisible chars that can cause phantom duplicates.
    """
    df = df.copy()

    # Normalize column names (strip & remove invisible characters)
    cols = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\u200b", "", regex=False)  # zero-width space
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
        .tolist()
    )

    counts = {}
    new_cols = []
    dupes = []

    for c in cols:
        if c in counts:
            counts[c] += 1
            dupes.append(c)
            new_cols.append(f"{c}__dup{counts[c]}")
        else:
            counts[c] = 0
            new_cols.append(c)

    df.columns = new_cols

    if dupes:
        st.warning(
            f"Duplicate columns detected{(' in ' + context) if context else ''}: "
            f"{sorted(set(dupes))}. Renamed duplicates with __dup# suffix."
        )

    return df


def assert_unique_columns(df: pd.DataFrame, name: str):
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        st.error(f"[{name}] Duplicate columns still present: {dupes}")
        st.write("Columns:", cols)
        st.stop()


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
