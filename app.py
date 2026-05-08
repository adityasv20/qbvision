import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os
import urllib.parse

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from qbvision.chart_builder import render_chart_builder
from qbvision.utils import force_unique_columns, assert_unique_columns, coerce_numeric


RATINGS_PATH = "data/qbvision_ratings.csv"
MODEL_PATH = "data/qb_model.joblib"

st.set_page_config(page_title="QBVision", page_icon="", layout="wide")


_football_svg = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='340' height='340'>"
    # football 1 — lower-left, tilted left
    "<g transform='translate(85,100) rotate(-30)' opacity='0.18'>"
    "<ellipse cx='0' cy='0' rx='56' ry='31' fill='none' stroke='#C8845A' stroke-width='2.5'/>"
    "<path d='M-52,0 Q-26,-28 0,-28 Q26,-28 52,0' fill='none' stroke='#9A5C38' stroke-width='1.2'/>"
    "<path d='M-52,0 Q-26,28 0,28 Q26,28 52,0' fill='none' stroke='#9A5C38' stroke-width='1.2'/>"
    "<line x1='0' y1='-26' x2='0' y2='26' stroke='#E8B088' stroke-width='1.8'/>"
    "<line x1='-11' y1='-13' x2='11' y2='-13' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='-6' x2='11' y2='-6' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='1' x2='11' y2='1' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='8' x2='11' y2='8' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='15' x2='11' y2='15' stroke='#E8B088' stroke-width='1.4'/>"
    "</g>"
    # football 2 — upper-right, tilted right
    "<g transform='translate(248,220) rotate(22)' opacity='0.18'>"
    "<ellipse cx='0' cy='0' rx='56' ry='31' fill='none' stroke='#C8845A' stroke-width='2.5'/>"
    "<path d='M-52,0 Q-26,-28 0,-28 Q26,-28 52,0' fill='none' stroke='#9A5C38' stroke-width='1.2'/>"
    "<path d='M-52,0 Q-26,28 0,28 Q26,28 52,0' fill='none' stroke='#9A5C38' stroke-width='1.2'/>"
    "<line x1='0' y1='-26' x2='0' y2='26' stroke='#E8B088' stroke-width='1.8'/>"
    "<line x1='-11' y1='-13' x2='11' y2='-13' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='-6' x2='11' y2='-6' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='1' x2='11' y2='1' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='8' x2='11' y2='8' stroke='#E8B088' stroke-width='1.4'/>"
    "<line x1='-11' y1='15' x2='11' y2='15' stroke='#E8B088' stroke-width='1.4'/>"
    "</g>"
    # football 3 — top-center, steep angle, smaller/dimmer
    "<g transform='translate(185,52) rotate(55)' opacity='0.10'>"
    "<ellipse cx='0' cy='0' rx='42' ry='23' fill='none' stroke='#C8845A' stroke-width='2'/>"
    "<path d='M-39,0 Q-20,-20 0,-20 Q20,-20 39,0' fill='none' stroke='#9A5C38' stroke-width='1'/>"
    "<path d='M-39,0 Q-20,20 0,20 Q20,20 39,0' fill='none' stroke='#9A5C38' stroke-width='1'/>"
    "<line x1='0' y1='-19' x2='0' y2='19' stroke='#E8B088' stroke-width='1.5'/>"
    "<line x1='-8' y1='-9' x2='8' y2='-9' stroke='#E8B088' stroke-width='1.1'/>"
    "<line x1='-8' y1='-3' x2='8' y2='-3' stroke='#E8B088' stroke-width='1.1'/>"
    "<line x1='-8' y1='3' x2='8' y2='3' stroke='#E8B088' stroke-width='1.1'/>"
    "<line x1='-8' y1='9' x2='8' y2='9' stroke='#E8B088' stroke-width='1.1'/>"
    "</g>"
    "</svg>"
)
_bg_url = urllib.parse.quote(_football_svg, safe="")

CUSTOM_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{
  background-color: #0d1117;
  background-image: url("data:image/svg+xml,{_bg_url}");
  background-size: 340px 340px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: rgba(13,17,23,0.92); }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
small {{ opacity: 0.8; }}

.qbv-card {{
  width: 100%;
  box-sizing: border-box;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.18);
  max-width: 100%;
  overflow-wrap: anywhere;
}}

.qbv-divider {{
  height: 1px;
  background: rgba(255,255,255,0.10);
  margin: 14px 0;
}}

[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }}
.js-plotly-plot, .plot-container {{ border-radius: 14px; }}

.qbv-pill {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  margin-right: 6px;
}}

.qbv-caption {{ opacity: 0.82; font-size: 13px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(RATINGS_PATH)

    # ensure plotly-safe cols immediately
    df = force_unique_columns(df, context="load_data after read_csv")
    assert_unique_columns(df, "load_data (post-fix)")

    num_cols = [
        "dropbacks", "epa_per_db", "context_pred_epa_per_db", "skill_residual",
        "team_sack_rate_allowed", "team_yac_per_comp", "team_no_huddle_rate",
        "int_rate", "sack_rate", "ay_per_att", "yac_per_comp",
        "QBVision_Rating", "next_QBVision_Rating",
        "efficiency_score", "decision_score", "explosiveness_score", "mobility_score"
    ]
    df = coerce_numeric(df, num_cols)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season", "qb_name", "team", "QBVision_Rating"]).copy()

    # enforce again after transformations
    df = force_unique_columns(df, context="load_data after cleaning")
    assert_unique_columns(df, "load_data after cleaning (post-fix)")

    return df


def percentile_within_season(season_df: pd.DataFrame, col: str, value: float) -> float:
    s = season_df[col].dropna().values
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s < value).mean())


def badge_for_rating(r):
    if pd.isna(r): return "—"
    if r >= 90: return "Elite"
    if r >= 80: return "All-Pro"
    if r >= 70: return "Great"
    if r >= 60: return "Above Avg"
    if r >= 50: return "Average"
    return "Developing"


def make_radar(categories, values, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Percentiles"
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10))),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        height=360,
    )
    return fig


# Load model + data
df = load_data()
df = force_unique_columns(df, context="app.py df after load_data")
assert_unique_columns(df, "app.py df after load_data (post-fix)")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
MODEL_FEATURES = bundle["features"]
BLEND_M = bundle["pred_blend_model"]
BLEND_C = bundle["pred_blend_current"]

latest_season = int(df["season"].max())
min_season = int(df["season"].min())


# Header
st.markdown("""
<div style="text-align:center; padding: 2.8rem 0 2rem 0;">
  <div style="display:inline-flex; align-items:center; gap:16px; justify-content:center;">
    <svg xmlns="http://www.w3.org/2000/svg" width="58" height="34" viewBox="0 0 58 34">
      <ellipse cx="29" cy="17" rx="27" ry="15" fill="#7A3A18" stroke="#C8845A" stroke-width="1.8"/>
      <path d="M 4 17 Q 16 4 29 4 Q 42 4 54 17" fill="none" stroke="#9A5C30" stroke-width="1.2"/>
      <path d="M 4 17 Q 16 30 29 30 Q 42 30 54 17" fill="none" stroke="#9A5C30" stroke-width="1.2"/>
      <line x1="29" y1="5" x2="29" y2="29" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="22" y1="10" x2="36" y2="10" stroke="white" stroke-width="1.4" stroke-linecap="round"/>
      <line x1="22" y1="14" x2="36" y2="14" stroke="white" stroke-width="1.4" stroke-linecap="round"/>
      <line x1="22" y1="18" x2="36" y2="18" stroke="white" stroke-width="1.4" stroke-linecap="round"/>
      <line x1="22" y1="22" x2="36" y2="22" stroke="white" stroke-width="1.4" stroke-linecap="round"/>
    </svg>
    <span style="font-size:54px; font-weight:900; letter-spacing:-0.03em; background:linear-gradient(135deg,#ffffff 20%,#e0a87a 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1;">QBVision</span>
  </div>
  <div style="margin-top:10px; font-size:12px; opacity:0.45; letter-spacing:0.12em; text-transform:uppercase; font-weight:500;">Context-aware quarterback rating &amp; projection system</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='qbv-divider'></div>", unsafe_allow_html=True)


# Global filters
filter_row = st.columns([0.34, 0.33, 0.33])
with filter_row[0]:
    season = st.selectbox(
        "Season",
        sorted(df["season"].unique().astype(int)),
        index=len(sorted(df["season"].unique().astype(int))) - 1
    )
with filter_row[1]:
    min_db = st.slider("Minimum dropbacks", 50, 600, 150, step=25)
with filter_row[2]:
    team_filter = st.multiselect("Team filter (optional)", sorted(df["team"].unique()), default=[])

season_df = df[(df["season"] == season) & (df["dropbacks"] >= min_db)].copy()
if team_filter:
    season_df = season_df[season_df["team"].isin(team_filter)].copy()

# hard guard before any plotting
season_df = force_unique_columns(season_df, context="season_df before tabs")
assert_unique_columns(season_df, "season_df before tabs (post-fix)")


# Tabs
tab1, tab2, tab3, tab4, tab_builder = st.tabs(
    ["Rankings", "QB Profile", "Projections", "Methodology", "Chart Builder"]
)


# Rankings Tab
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    best = season_df.sort_values("QBVision_Rating", ascending=False).head(1)
    if len(best) == 1:
        b = best.iloc[0]
        with c1:
            st.markdown(
                "<div class='qbv-card'>"
                "<div style='font-size:12px; opacity:0.8;'>QB1 (QBVision)</div>"
                f"<div style='font-size:20px; font-weight:800;'>{b['qb_name']}</div>"
                f"<div style='font-size:14px; opacity:0.85;'>{b['team']} · {badge_for_rating(b['QBVision_Rating'])}</div>"
                f"<div style='margin-top:10px; font-size:30px; font-weight:900;'>{b['QBVision_Rating']:.1f}</div>"
                "<div style='font-size:12px; opacity:0.75;'>QBVision Rating</div>"
                "</div>",
                unsafe_allow_html=True
            )

    with c2:
        st.markdown(
            "<div class='qbv-card'>"
            "<div style='font-size:12px; opacity:0.8;'>Season spread</div>"
            f"<div style='font-size:28px; font-weight:900;'>{season_df['QBVision_Rating'].mean():.1f}</div>"
            "<div style='font-size:12px; opacity:0.75;'>Average rating</div>"
            f"<div style='margin-top:8px; font-size:14px; opacity:0.85;'>Std dev: {season_df['QBVision_Rating'].std(ddof=0):.1f}</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            "<div class='qbv-card'>"
            "<div style='font-size:12px; opacity:0.8;'>Context effect</div>"
            f"<div style='font-size:28px; font-weight:900;'>{season_df['skill_residual'].mean():.3f}</div>"
            "<div style='font-size:12px; opacity:0.75;'>Avg residual (EPA/DB)</div>"
            f"<div style='margin-top:8px; font-size:14px; opacity:0.85;'>Higher = outperforming situation</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            "<div class='qbv-card'>"
            "<div style='font-size:12px; opacity:0.8;'>Sample</div>"
            f"<div style='font-size:28px; font-weight:900;'>{len(season_df)}</div>"
            "<div style='font-size:12px; opacity:0.75;'>Qualified QBs</div>"
            f"<div style='margin-top:8px; font-size:14px; opacity:0.85;'>Min DB: {min_db}</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='qbv-divider'></div>", unsafe_allow_html=True)

    show_cols = [
        "qb_name", "team", "dropbacks",
        "QBVision_Rating", "epa_per_db", "skill_residual",
        "int_rate", "sack_rate",
        "team_sack_rate_allowed", "team_yac_per_comp", "team_no_huddle_rate",
    ]
    show_cols = [c for c in show_cols if c in season_df.columns]
    board = season_df.sort_values("QBVision_Rating", ascending=False)[show_cols].copy()

    if "QBVision_Rating" in board.columns:
        board["QBVision_Rating"] = board["QBVision_Rating"].round(1)
    for c in ["epa_per_db", "skill_residual"]:
        if c in board.columns:
            board[c] = board[c].round(3)
    for c in ["int_rate", "sack_rate", "team_sack_rate_allowed", "team_no_huddle_rate"]:
        if c in board.columns:
            board[c] = (board[c] * 100).round(2)
    if "team_yac_per_comp" in board.columns:
        board["team_yac_per_comp"] = board["team_yac_per_comp"].round(2)

    st.subheader("Rankings")
    st.caption("Rates shown in % where applicable (INT%, Sack%, Team Sack Allowed%, No-huddle%).")
    st.dataframe(board, use_container_width=True, height=380)

    st.markdown("<div class='qbv-divider'></div>", unsafe_allow_html=True)

    # hard guard right before plotly calls
    season_df_plot = force_unique_columns(season_df, context="season_df rankings plots")
    assert_unique_columns(season_df_plot, "season_df rankings plots (post-fix)")

    v1, v2 = st.columns(2)
    with v1:
        fig = px.scatter(
            season_df_plot,
            x="team_sack_rate_allowed",
            y="epa_per_db",
            hover_name="qb_name",
            color="team",
            size="dropbacks",
            title="Efficiency vs Protection",
            labels={"team_sack_rate_allowed": "Team sack rate allowed (lower is better)", "epa_per_db": "EPA per dropback"}
        )
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=380)
        st.plotly_chart(fig, width="stretch")

    with v2:
        fig = px.scatter(
            season_df_plot,
            x="team_yac_per_comp",
            y="epa_per_db",
            hover_name="qb_name",
            color="team",
            size="dropbacks",
            title="Efficiency vs Weapons (YAC environment)",
            labels={"team_yac_per_comp": "Team YAC per completion", "epa_per_db": "EPA per dropback"}
        )
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=380)
        st.plotly_chart(fig, width="stretch")


# QB Profile Tab
with tab2:
    if len(season_df) == 0:
        st.info("No QBs match the current filters.")
    else:
        qbs = season_df.sort_values("QBVision_Rating", ascending=False)["qb_name"].unique().tolist()
        pick = st.selectbox("Select a quarterback", qbs, index=0)
        qb = season_df[season_df["qb_name"] == pick].iloc[0]

        p1, p2 = st.columns([0.62, 0.38], vertical_alignment="top")

        with p1:
            st.markdown(
                "<div class='qbv-card'>"
                f"<div style='font-size:12px; opacity:0.8;'>QB Profile · {season}</div>"
                f"<div style='font-size:26px; font-weight:900; margin-top:2px;'>{qb['qb_name']}</div>"
                f"<div style='font-size:14px; opacity:0.85;'>{qb['team']} · {badge_for_rating(qb['QBVision_Rating'])} · {int(qb['dropbacks'])} dropbacks</div>"
                "<div class='qbv-divider'></div>"
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("QBVision Rating", f"{qb['QBVision_Rating']:.1f}")
            m2.metric("EPA / DB", f"{qb['epa_per_db']:.3f}")
            m3.metric("Skill residual", f"{qb['skill_residual']:.3f}")
            m4.metric("INT rate", f"{qb['int_rate'] * 100:.2f}%")

        with p2:
            peers = season_df.copy()

            pct_rating = percentile_within_season(peers, "QBVision_Rating", qb["QBVision_Rating"]) * 100
            pct_skill = percentile_within_season(peers, "skill_residual", qb["skill_residual"]) * 100
            pct_eff = percentile_within_season(peers, "epa_per_db", qb["epa_per_db"]) * 100
            pct_int = (1 - percentile_within_season(peers, "int_rate", qb["int_rate"])) * 100
            pct_sack = (1 - percentile_within_season(peers, "sack_rate", qb["sack_rate"])) * 100

            st.markdown(
                "<div class='qbv-card'>"
                "<div style='font-size:12px; opacity:0.8;'>Percentiles vs qualified QBs</div>"
                f"<div style='margin-top:10px; font-size:14px;'>QBVision: <b>{pct_rating:.0f}th</b></div>"
                f"<div style='font-size:14px;'>Skill: <b>{pct_skill:.0f}th</b></div>"
                f"<div style='font-size:14px;'>Efficiency: <b>{pct_eff:.0f}th</b></div>"
                f"<div style='font-size:14px;'>INT avoidance: <b>{pct_int:.0f}th</b></div>"
                f"<div style='font-size:14px;'>Sack avoidance: <b>{pct_sack:.0f}th</b></div>"
                "<div class='qbv-divider'></div>"
                "<div style='font-size:12px; opacity:0.75;'>These percentiles are season-relative and use the current filters.</div>"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='qbv-divider'></div>", unsafe_allow_html=True)

        radar_categories = []
        radar_values = []

        subs = [
            ("Skill (residual)", "skill_residual", False),
            ("Efficiency (EPA/DB)", "epa_per_db", False),
            ("Decision (INT%)", "int_rate", True),
            ("Decision (Sack%)", "sack_rate", True),
            ("Explosive (AirYds/Att)", "ay_per_att", False),
            ("Weapons env (YAC/Comp)", "team_yac_per_comp", False),
            ("Protection env (SackAllowed)", "team_sack_rate_allowed", True),
            ("Tempo (No-huddle)", "team_no_huddle_rate", False),
        ]

        for label, col, invert in subs:
            if col not in peers.columns or pd.isna(qb[col]):
                continue
            p = percentile_within_season(peers, col, qb[col])
            if invert:
                p = 1 - p
            radar_categories.append(label)
            radar_values.append(p * 100)

        r1, r2 = st.columns([0.55, 0.45], vertical_alignment="top")
        with r1:
            fig = make_radar(radar_categories, radar_values, title="QBVision profile radar (percentiles)")
            st.plotly_chart(fig, width="stretch")

        with r2:
            st.markdown(
                "<div class='qbv-card'>"
                "<div style='font-size:12px; opacity:0.8;'>Context snapshot</div>"
                "<div style='margin-top:10px; font-size:14px;'>"
                f"Protection (team sack allowed): <b>{qb['team_sack_rate_allowed'] * 100:.2f}%</b><br/>"
                f"Weapons (team YAC/comp): <b>{qb['team_yac_per_comp']:.2f}</b><br/>"
                f"Tempo (no-huddle rate): <b>{qb['team_no_huddle_rate'] * 100:.1f}%</b>"
                "</div>"
                "<div class='qbv-divider'></div>"
                "<div style='font-size:12px; opacity:0.75;'>Context is used to adjust skill via a regression baseline.</div>"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='qbv-divider'></div>", unsafe_allow_html=True)

        if "context_pred_epa_per_db" in season_df.columns:
            comp = season_df.copy()
            comp = force_unique_columns(comp, context="qb profile comp")
            comp["over_expected"] = comp["epa_per_db"] - comp["context_pred_epa_per_db"]

            fig = px.scatter(
                comp,
                x="context_pred_epa_per_db",
                y="epa_per_db",
                hover_name="qb_name",
                color="team",
                size="dropbacks",
                title="Actual EPA/DB vs Context Baseline",
                labels={"context_pred_epa_per_db": "Context-predicted EPA/DB", "epa_per_db": "Actual EPA/DB"}
            )
            fig.add_trace(go.Scatter(
                x=[qb["context_pred_epa_per_db"]],
                y=[qb["epa_per_db"]],
                mode="markers",
                marker=dict(size=18, symbol="star"),
                name="Selected QB"
            ))
            fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
            st.plotly_chart(fig, width="stretch")


# Projections Tab
with tab3:
    if len(season_df) == 0:
        st.info("No QBs match the current filters.")
    else:
        st.subheader("Next-season QBVision projections")
        st.caption("Predictions are generated from a model trained on season-to-season transitions using QBVision components and context features.")

        missing = [c for c in MODEL_FEATURES if c not in season_df.columns]
        if missing:
            st.error(f"Your model expects columns that are missing from qbvision_ratings.csv: {missing}")
        else:
            X = season_df[MODEL_FEATURES].copy()

            season_df_proj = season_df.copy()
            season_df_proj = force_unique_columns(season_df_proj, context="season_df_proj before predict")

            pred = model.predict(X)
            pred = BLEND_M * pred + BLEND_C * season_df_proj["QBVision_Rating"].values
            season_df_proj["pred_next_qbvision"] = pred

            topN = st.slider("Show top N", 5, 30, 12)
            proj = season_df_proj.sort_values("pred_next_qbvision", ascending=False).head(topN).copy()

            c1, c2 = st.columns([0.55, 0.45], vertical_alignment="top")
            with c1:
                fig = px.bar(
                    proj.sort_values("pred_next_qbvision"),
                    x="pred_next_qbvision",
                    y="qb_name",
                    orientation="h",
                    title="Top projected QBs next season (QBVision Rating)",
                    labels={"pred_next_qbvision": "Predicted next-season QBVision", "qb_name": "QB"}
                )
                fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
                st.plotly_chart(fig, width="stretch")

            with c2:
                season_df_proj["delta"] = season_df_proj["pred_next_qbvision"] - season_df_proj["QBVision_Rating"]
                risers = season_df_proj.sort_values("delta", ascending=False).head(8)[["qb_name", "team", "QBVision_Rating", "pred_next_qbvision", "delta"]]
                fallers = season_df_proj.sort_values("delta", ascending=True).head(8)[["qb_name", "team", "QBVision_Rating", "pred_next_qbvision", "delta"]]

                st.markdown("<div class='qbv-card'>", unsafe_allow_html=True)
                st.markdown("**Biggest risers (model vs this season)**")
                st.dataframe(
                    risers.assign(
                        QBVision_Rating=lambda d: d["QBVision_Rating"].round(1),
                        pred_next_qbvision=lambda d: d["pred_next_qbvision"].round(1),
                        delta=lambda d: d["delta"].round(1),
                    ),
                    use_container_width=True,
                    height=220
                )
                st.markdown("**Biggest fallers (model vs this season)**")
                st.dataframe(
                    fallers.assign(
                        QBVision_Rating=lambda d: d["QBVision_Rating"].round(1),
                        pred_next_qbvision=lambda d: d["pred_next_qbvision"].round(1),
                        delta=lambda d: d["delta"].round(1),
                    ),
                    use_container_width=True,
                    height=220
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='qbv-divider'></div>", unsafe_allow_html=True)

            fig = px.scatter(
                season_df_proj,
                x="QBVision_Rating",
                y="pred_next_qbvision",
                hover_name="qb_name",
                color="team",
                size="dropbacks",
                title="Current QBVision vs Predicted Next Season",
                labels={"QBVision_Rating": "Current QBVision", "pred_next_qbvision": "Predicted next-season QBVision"}
            )
            fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
            st.plotly_chart(fig, width="stretch")


# Methodology Tab
with tab4:
    st.subheader("QBVision Methodology")
    st.markdown(
        """
### Overview
QBVision is a **season-relative quarterback evaluation system** built from play-by-play signals.  
It aims to measure **true quarterback skill** while accounting for offensive environment.

---

### 1) Core quarterback signals
Each QB season is summarized using dropback-level metrics:
- **EPA per dropback** (efficiency)
- **Interception rate** (decision risk)
- **Sack rate** (pressure response)
- **Air yards per attempt** (aggressiveness)
- **YAC per completion** (playmaking environment)
- **Dropbacks** (sample stability)

---

### 2) Quantifying environment (context)
QBVision approximates supporting cast and structure using team-level aggregates:
- **Protection:** team sack rate allowed  
- **Weapons:** team YAC per completion  
- **Tempo / scheme proxy:** **no-huddle rate**  

---

### 3) Context adjustment (skill residual)
We fit a baseline model:
> EPA/DB ≈ f(protection, weapons, tempo)

Then compute:
> **Skill residual = actual EPA/DB − context-expected EPA/DB**

---

### 4) Final QBVision Rating (0–100)
Metrics are normalized within each season, then blended into a score:
- context-adjusted skill
- raw efficiency
- decision-making
- explosiveness proxies

---

### 5) Next-season projections (ML)
A regression model trained on season-to-season transitions predicts next-season performance.
        """
    )


# Chart Builder Tab
with tab_builder:
    render_chart_builder(df)
