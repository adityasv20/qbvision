import json
import base64
import plotly.express as px
import streamlit as st

from qbvision.metrics_catalog import metric_catalog
from qbvision.utils import force_unique_columns, assert_unique_columns


def _b64_encode_json(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _b64_decode_json(s: str):
    try:
        raw = base64.urlsafe_b64decode(s.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def render_chart_builder(df):
    # hard guard: fix duplicates at entry
    df = force_unique_columns(df, context="chart builder input")
    assert_unique_columns(df, "chart builder df (post-fix)")

    st.subheader("🎛️ Build Your Own Chart")

    required_cols = {"season", "qb_name", "team"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Chart Builder expects columns: {missing}. Check your qbvision_ratings.csv schema.")
        return

    catalog = [m for m in metric_catalog() if m["key"] in df.columns]
    if len(catalog) < 2:
        st.warning("Not enough metrics found. Add columns or update metric_catalog().")
        return

    labels = [m["label"] for m in catalog]
    label_to_key = {m["label"]: m["key"] for m in catalog}

    qp = st.query_params
    cfg_from_url = _b64_decode_json(qp["chart"]) if "chart" in qp else None

    seasons = sorted(df["season"].dropna().unique().tolist())
    min_season, max_season = (min(seasons), max(seasons)) if seasons else (2018, 2025)

    default_cfg = {
        "chart_type": "Scatter",
        "x": "EPA per Dropback",
        "y": "QBVision Rating",
        "color": "Team",
        "size": "Dropbacks",
        "trendline": True,
        "season_range": [min_season, max_season],
        "min_dropbacks": 200,
        "teams": [],
        "qbs": [],
        "facet": "None",
    }
    cfg = cfg_from_url or default_cfg

    left, right = st.columns([1, 1])
    with left:
        chart_type = st.selectbox(
            "Chart type",
            ["Scatter", "Line (Trend Over Time)", "Bar (Rank / Compare)", "Histogram"],
            index=["Scatter", "Line (Trend Over Time)", "Bar (Rank / Compare)", "Histogram"].index(cfg.get("chart_type", "Scatter"))
            if cfg.get("chart_type") in ["Scatter", "Line (Trend Over Time)", "Bar (Rank / Compare)", "Histogram"]
            else 0,
        )
        x_label = st.selectbox("X-axis metric", labels, index=labels.index(cfg["x"]) if cfg.get("x") in labels else 0)
        y_label = st.selectbox("Y-axis metric", labels, index=labels.index(cfg["y"]) if cfg.get("y") in labels else 1)
        trendline = st.checkbox("Add trendline (scatter only)", value=bool(cfg.get("trendline", True)))

    with right:
        color_opt = st.selectbox(
            "Color by",
            ["None", "Team", "Season"],
            index=["None", "Team", "Season"].index(cfg.get("color", "Team"))
            if cfg.get("color") in ["None", "Team", "Season"]
            else 1,
        )
        size_opt = st.selectbox(
            "Size by (scatter only)",
            ["None"] + labels,
            index=(["None"] + labels).index(cfg.get("size", "Dropbacks"))
            if cfg.get("size") in (["None"] + labels)
            else 0,
        )
        facet_opt = st.selectbox(
            "Facet",
            ["None", "Season", "Team"],
            index=["None", "Season", "Team"].index(cfg.get("facet", "None"))
            if cfg.get("facet") in ["None", "Season", "Team"]
            else 0,
        )
        season_range = st.slider(
            "Season range",
            min_value=min_season,
            max_value=max_season,
            value=tuple(cfg.get("season_range", [min_season, max_season])),
            step=1,
        )

    st.markdown("---")

    teams_all = sorted(df["team"].dropna().unique().tolist())
    qbs_all = sorted(df["qb_name"].dropna().unique().tolist())

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        min_dropbacks = st.number_input("Min dropbacks", min_value=0, max_value=2000, value=int(cfg.get("min_dropbacks", 200)), step=25)
    with c2:
        teams_sel = st.multiselect("Filter teams (optional)", teams_all, default=[t for t in cfg.get("teams", []) if t in teams_all])
    with c3:
        qbs_sel = st.multiselect("Filter QBs (optional)", qbs_all, default=[q for q in cfg.get("qbs", []) if q in qbs_all])

    dff = df.copy()
    dff = dff[(dff["season"] >= season_range[0]) & (dff["season"] <= season_range[1])]
    if "dropbacks" in dff.columns:
        dff = dff[dff["dropbacks"] >= min_dropbacks]
    if teams_sel:
        dff = dff[dff["team"].isin(teams_sel)]
    if qbs_sel:
        dff = dff[dff["qb_name"].isin(qbs_sel)]

    # hard guard again after filtering
    dff = force_unique_columns(dff, context="chart builder filtered")
    assert_unique_columns(dff, "chart builder filtered (post-fix)")

    x_key = label_to_key[x_label]
    y_key = label_to_key[y_label]

    if x_key not in dff.columns or y_key not in dff.columns:
        st.error("Selected metrics not available after filtering.")
        return

    dff = dff.dropna(subset=[x_key, y_key])
    if len(dff) == 0:
        st.info("No rows match your filters. Try lowering min dropbacks or widening season range.")
        return

    hover_cols = ["qb_name", "team", "season"]
    if "dropbacks" in dff.columns:
        hover_cols.append("dropbacks")

    # avoid repeating keys in hover_data list
    for k in [x_key, y_key]:
        if k not in hover_cols:
            hover_cols.append(k)

    color_col = None if color_opt == "None" else ("team" if color_opt == "Team" else "season")
    facet_col = None if facet_opt == "None" else ("season" if facet_opt == "Season" else "team")

    if chart_type == "Scatter":
        size_col = None if size_opt == "None" else label_to_key.get(size_opt)
        if size_col is not None and size_col not in dff.columns:
            size_col = None

        fig = px.scatter(
            dff,
            x=x_key,
            y=y_key,
            color=color_col,
            size=size_col,
            trendline="ols" if (trendline and len(dff) >= 10) else None,
            facet_col=facet_col,
            hover_data=hover_cols,
            labels={x_key: x_label, y_key: y_label},
        )

    elif chart_type == "Line (Trend Over Time)":
        # if user doesn't choose QBs, keep it readable
        if not qbs_sel and "dropbacks" in dff.columns:
            top_qbs = (
                dff.groupby("qb_name")["dropbacks"].sum()
                .sort_values(ascending=False)
                .head(6)
                .index.tolist()
            )
            dff = dff[dff["qb_name"].isin(top_qbs)]

        fig = px.line(
            dff.sort_values("season"),
            x="season",
            y=y_key,
            color="qb_name",
            markers=True,
            hover_data=hover_cols,
            labels={y_key: y_label},
        )
        fig.update_xaxes(dtick=1)

    elif chart_type == "Bar (Rank / Compare)":
        latest = int(dff["season"].max()) if len(dff) else season_range[1]
        dff_latest = dff[dff["season"] == latest].copy()
        if qbs_sel:
            dff_latest = dff_latest[dff_latest["qb_name"].isin(qbs_sel)]
        else:
            dff_latest = dff_latest.sort_values(y_key, ascending=False).head(20)

        if len(dff_latest) == 0:
            st.info("No rows match your filters for the latest season in range.")
            return

        fig = px.bar(
            dff_latest.sort_values(y_key, ascending=True),
            x=y_key,
            y="qb_name",
            color=color_col,
            orientation="h",
            hover_data=hover_cols,
            labels={y_key: y_label},
        )

    else:  # Histogram
        fig = px.histogram(
            dff,
            x=x_key,
            color=color_col,
            nbins=30,
            hover_data=hover_cols,
            labels={x_key: x_label},
        )

    fig.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")

    cfg_out = {
        "chart_type": chart_type,
        "x": x_label,
        "y": y_label,
        "color": color_opt,
        "size": size_opt,
        "trendline": trendline,
        "season_range": list(season_range),
        "min_dropbacks": int(min_dropbacks),
        "teams": teams_sel,
        "qbs": qbs_sel,
        "facet": facet_opt,
    }

    st.caption("Share this exact chart view:")
    st.code(f"?chart={_b64_encode_json(cfg_out)}", language="text")

    st.download_button(
        "Download filtered data (CSV)",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name="qbvision_chart_data.csv",
        mime="text/csv",
    )
