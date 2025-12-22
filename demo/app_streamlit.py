import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium
from shapely.geometry import shape, Point
import geopandas as gpd
from aoi_map import aoi_draw_map

from dummy_data import (
    make_dummy_grid, make_dummy_forest, make_dummy_weather,
    make_cap_warning, apply_dummy_nlp_to_grid, make_dummy_finland_cells
)
from demo_pipeline import (
    fuse_features, train_demo_model, predict_with_model,
    estimate_uncertainty, add_explanations
)

# Optional click support
try:
    from streamlit_plotly_events import plotly_events
    HAS_CLICK = True
except Exception:
    HAS_CLICK = False

st.set_page_config(page_title="Windthrow Nowcasting Demo", layout="wide")

RESULTS_DIR = "results/demo"
os.makedirs(RESULTS_DIR, exist_ok=True)

@st.cache_data
def build_base_data(n_cells: int = 1200, hours: int = 24):
    """
    Build dummy Finland-shaped demo data.

    Parameters
    ----------
    n_cells : int
        Number of dummy 'grid cells' (centroids) distributed inside Finland.
        (For the slider 'n x n', call with n_cells=n*n.)
    hours : int
        Forecast horizon in hours.

    Returns
    -------
    grid : DataFrame with columns [cell_id, lat, lon]
    forest : DataFrame with columns [cell_id, spruce_pct, stand_age, avg_height, soil_type]
    weather : DataFrame with columns [cell_id, valid_time, gust10, wspd10, wdir10, prcp_conv, mslp]
    times : sorted list of forecast timestamps
    """
    grid = make_dummy_finland_cells(n_cells=n_cells)

    # forest + weather generators MUST work with cell_id and lat/lon
    forest = make_dummy_forest(grid)

    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    weather = make_dummy_weather(grid, start_time=start, hours=hours)

    times = sorted(weather["valid_time"].unique())
    return grid, forest, weather, times

def run_pipeline(grid, forest, weather, cap_sev):
    cap = make_cap_warning(cap_sev)
    nlp = apply_dummy_nlp_to_grid(grid, cap)
    fused = fuse_features(weather, forest, nlp)

    model, auc = train_demo_model(fused)
    pred = predict_with_model(model, fused)
    pred = estimate_uncertainty(pred)
    pred = add_explanations(pred)

    return pred, auc, cap

# def grid_heatmap(grid_df, df_t, value_col, title):
#     n = int(np.sqrt(len(grid_df)))
#     merged = df_t.merge(grid_df[["cell_id", "x", "y"]], on="cell_id", how="left")
#     img = np.full((n, n), np.nan, dtype=float)
#     for _, r in merged.iterrows():
#         img[int(r["y"]), int(r["x"])] = float(r[value_col])
#
#     fig = px.imshow(
#         img,
#         origin="lower",
#         aspect="equal",
#         title=title,
#         labels={"x": "X", "y": "Y", "color": value_col}
#     )
#     fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
#     return fig

# ---------------- UI ----------------
st.title("🌲 Windthrow Nowcasting Demo (Dummy Data + LightGBM + MAS Triggers)")

with st.sidebar:
    st.header("Controls")
    cap_sev = st.selectbox("CAP Warning Severity", ["minor", "moderate", "severe"], index=1)
    hours = st.slider("Forecast horizon (hours)", 6, 48, 24, 6)
    n = st.slider("Grid size (n x n)", 20, 60, 30, 5)

    trigger = st.button("🚨 Simulate NEW CAP Warning (Trigger Update)")

grid, forest, weather, times = build_base_data(n_cells=n*n, hours=hours)

# Trigger simulation: if pressed, cycle severity
if "cap_state" not in st.session_state:
    st.session_state.cap_state = cap_sev
if trigger:
    cycle = {"minor": "moderate", "moderate": "severe", "severe": "minor"}
    st.session_state.cap_state = cycle.get(st.session_state.cap_state, "moderate")
else:
    st.session_state.cap_state = cap_sev

cap_sev_effective = st.session_state.cap_state

pred, auc, cap_obj = run_pipeline(grid, forest, weather, cap_sev_effective)

# Time slider
time_options = sorted(pred["valid_time"].unique())
t_idx = st.slider("Select forecast time", 0, len(time_options) - 1, 0)
t_sel = time_options[t_idx]

df_t = pred[pred["valid_time"] == t_sel].copy()

# --- Finland overview (national view) ---------------------------------------
st.subheader("Finland-level Windthrow Risk Overview")
st.caption(
    f"CAP severity: **{cap_obj['cap_severity']}** | "
    f"Model AUC (synthetic): **{auc:.3f}** | "
    f"Time: **{t_sel}**"
)

# Merge lat/lon into df_t (grid must have lat/lon columns)
df_map = df_t.merge(grid[["cell_id", "lat", "lon"]], on="cell_id", how="left")

fig_fin = px.scatter_mapbox(
    df_map,
    lat="lat",
    lon="lon",
    color="risk_prob",
    size="risk_prob",
    hover_data={
        "cell_id": True,
        "severity_class": True,
        "risk_uncertainty": ":.2f",
        "risk_prob": ":.2f",
    },
    zoom=4.5,
    mapbox_style="carto-positron",
    title="Windthrow risk probability (Finland overview)",
)
st.plotly_chart(fig_fin, use_container_width=True)

# --- AOI selection (draw tool) ----------------------------------------------
st.subheader("Select Area of Interest (AOI)")
m = aoi_draw_map()
draw_data = st_folium(m, height=420, width=None)

aoi_geom = None
if draw_data and draw_data.get("last_active_drawing"):
    aoi_geom = shape(draw_data["last_active_drawing"]["geometry"])

# Filter cells inside AOI
grid_in_aoi = grid.copy()
if aoi_geom is not None:
    pts_inside = []
    for _, r in grid.iterrows():
        if aoi_geom.contains(Point(r["lon"], r["lat"])):
            pts_inside.append(r["cell_id"])
    df_aoi = df_t[df_t["cell_id"].isin(pts_inside)].copy()
    grid_in_aoi = grid[grid["cell_id"].isin(pts_inside)].copy()

    st.success(f"AOI selected: showing {len(df_aoi)} cells inside AOI")
else:
    df_aoi = df_t.copy()
    st.info("Draw a rectangle/polygon on the map to zoom into a region.")

# --- Two-level view: AOI detailed grid + explanation panel -------------------
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader("AOI Detail View")

    # 1) AOI detail map (clickable if you want later)
    df_aoi_map = df_aoi.merge(grid[["cell_id", "lat", "lon"]], on="cell_id", how="left")
    zoom_lvl = 6 if aoi_geom is not None else 4.5

    fig_aoi = px.scatter_mapbox(
        df_aoi_map,
        lat="lat",
        lon="lon",
        color="risk_prob",
        size="risk_prob",
        hover_data={
            "cell_id": True,
            "severity_class": True,
            "risk_uncertainty": ":.2f",
            "risk_prob": ":.2f",
        },
        zoom=zoom_lvl,
        mapbox_style="carto-positron",
        title="AOI risk points (cells)",
    )
    st.plotly_chart(fig_aoi, use_container_width=True)

    # 2) AOI grid-level binned heatmap (only meaningful when AOI exists)
    if aoi_geom is not None and len(df_aoi_map) > 20:
        st.subheader("AOI Grid-level Risk (binned)")

        # Bin into a local grid for a "grid view" demo effect
        df_local = df_aoi_map.copy()
        df_local["bx"] = pd.cut(df_local["lon"], bins=30, labels=False)
        df_local["by"] = pd.cut(df_local["lat"], bins=30, labels=False)

        pivot = df_local.pivot_table(
            index="by", columns="bx", values="risk_prob", aggfunc="mean"
        )

        fig_detail = px.imshow(
            pivot.sort_index(ascending=False),
            title="Windthrow risk probability inside AOI (binned grid)",
            aspect="equal",
        )
        st.plotly_chart(fig_detail, use_container_width=True)

with c2:
    st.subheader("Cell Explanation")

    # Only allow selecting cells inside AOI (if AOI exists)
    candidate_cells = (
        grid_in_aoi["cell_id"].tolist() if aoi_geom is not None else grid["cell_id"].tolist()
    )

    if len(candidate_cells) == 0:
        st.warning("No cells found inside AOI. Draw a larger AOI.")
    else:
        sel_cell_id = int(
            st.selectbox("Pick a cell", candidate_cells, index=0)
        )

        row = df_t[df_t["cell_id"] == sel_cell_id].iloc[0]

        st.metric("Risk probability", f"{row['risk_prob']:.2f}")
        st.metric("Uncertainty", f"{row['risk_uncertainty']:.2f}")
        st.metric("Severity", row["severity_class"])

        st.write("**Explanation**")
        st.info(row["explanation"])

        st.write("**Key drivers (inputs)**")
        st.json({
            "gust10": float(row["gust10"]),
            "wspd10": float(row["wspd10"]),
            "wdir10": float(row["wdir10"]),
            "prcp_conv": float(row["prcp_conv"]),
            "mslp": float(row["mslp"]),
            "spruce_pct": float(row["spruce_pct"]),
            "stand_age": float(row["stand_age"]),
            "avg_height": float(row["avg_height"]),
            "soil_type": row["soil_type"],
            "cap_severity": row["cap_severity"],
            "treefall_flag": int(row["treefall_flag"]),
            "outage_flag": int(row["outage_flag"]),
            "road_flag": int(row["road_flag"]),
        })

# Save snapshot (optional)
if st.button("💾 Save current snapshot to results/demo"):
    snap_path = os.path.join(RESULTS_DIR, f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.parquet")
    pred.to_parquet(snap_path, index=False)
    st.success(f"Saved snapshot: {snap_path}")