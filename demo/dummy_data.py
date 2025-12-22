import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import geopandas as gpd
from shapely.geometry import Point
import os

def make_dummy_grid(n=30, lat0=63.0, lon0=25.0, d=0.03):
    # Dummy “Finland patch” lat/lon lattice
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    lat = lat0 + (ys - n/2) * d
    lon = lon0 + (xs - n/2) * d
    df = pd.DataFrame({
        "cell_id": np.arange(n*n),
        "x": xs.ravel(),
        "y": ys.ravel(),
        "lat": lat.ravel(),
        "lon": lon.ravel(),
    })
    return df

def load_finland_polygon():
    """
    Load Finland polygon from Natural Earth (GeoPandas >= 1.0 compatible).
    """
    shp_path = os.path.join("demo","natural_earth","ne_110m_admin_0_countries.shp")
    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Natural Earth shapefile not found at {shp_path}. "
            "Download it from https://www.naturalearthdata.com/"
        )

    world = gpd.read_file(shp_path)
    fin = world[world["NAME"] == "Finland"].to_crs(epsg=4326)

    if fin.empty:
        raise ValueError("Finland polygon not found in Natural Earth dataset.")

    return fin.geometry.iloc[0]

def make_dummy_finland_cells(n_cells=1200, seed=42):
    poly = load_finland_polygon()
    minx, miny, maxx, maxy = poly.bounds

    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_cells:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            pts.append(p)

    return pd.DataFrame({
        "cell_id": np.arange(n_cells),
        "lon": [p.x for p in pts],
        "lat": [p.y for p in pts],
    })

def make_dummy_forest(grid_df, seed=42):
    rng = np.random.default_rng(seed)
    n = len(grid_df)

    lat = grid_df["lat"].to_numpy()
    # north-south gradient (dummy): higher spruce in north
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)

    spruce_pct = np.clip(0.25 + 0.55 * lat_norm + rng.normal(0, 0.08, n), 0, 1)
    age = np.clip(15 + 90 * spruce_pct + rng.normal(0, 10, n), 10, 140)
    height = np.clip(4 + 28 * spruce_pct + rng.normal(0, 3, n), 3, 40)
    soil_type = rng.choice(["mineral", "peat"], size=n, p=[0.72, 0.28])

    return pd.DataFrame({
        "cell_id": grid_df["cell_id"].values,
        "spruce_pct": spruce_pct,
        "stand_age": age,
        "avg_height": height,
        "soil_type": soil_type,
    })

def make_cap_warning(severity: str):
    severity = severity.lower()
    texts = {
        "minor": "Strong winds possible. Caution outdoors.",
        "moderate": "Very strong gusts expected. Falling trees possible. Drive carefully.",
        "severe": "Storm-level gusts. High risk of treefall and power outages. Avoid travel."
    }
    if severity not in texts:
        raise ValueError("severity must be one of: minor, moderate, severe")
    return {"cap_severity": severity, "cap_text": texts[severity]}

def apply_dummy_nlp_to_grid(grid_df, cap_obj):
    # Convert CAP severity/text into grid features (for demo, same for all cells)
    sev = cap_obj["cap_severity"]
    sev_score = {"minor": 0.2, "moderate": 0.6, "severe": 0.9}[sev]

    text = cap_obj["cap_text"].lower()
    treefall = 1 if ("tree" in text or "falling" in text) else 0
    outage = 1 if ("outage" in text or "power" in text) else 0
    road = 1 if ("travel" in text or "drive" in text or "road" in text) else 0

    # “uncertainty language” dummy score
    unc = 0.2 if "possible" in text else 0.1
    if "expected" in text or "high risk" in text:
        unc = 0.05

    return pd.DataFrame({
        "cell_id": grid_df["cell_id"].values,
        "cap_severity": sev,
        "cap_sev_score": sev_score,
        "treefall_flag": treefall,
        "outage_flag": outage,
        "road_flag": road,
        "text_uncertainty": unc
    })

def make_dummy_weather(grid_df, start_time: datetime, hours=24, seed=123):
    rng = np.random.default_rng(seed)
    n = len(grid_df)
    times = [start_time + timedelta(hours=h) for h in range(hours)]

    lat = grid_df["lat"].to_numpy()
    lon = grid_df["lon"].to_numpy()

    # Storm center moves SW -> NE across Finland bounding box
    min_lat, max_lat = float(lat.min()), float(lat.max())
    min_lon, max_lon = float(lon.min()), float(lon.max())

    rows = []
    for h, t in enumerate(times):
        frac = h / max(hours - 1, 1)
        c_lat = min_lat + frac * (max_lat - min_lat)
        c_lon = min_lon + frac * (max_lon - min_lon)

        # distance in degrees (good enough for demo)
        dist = np.sqrt((lat - c_lat) ** 2 + (lon - c_lon) ** 2)
        storm = np.exp(-(dist**2) / (2 * (1.2 ** 2)))  # spread ~1.2 degrees

        gust10 = 11 + 20 * storm + rng.normal(0, 1.2, n)
        wspd10 = 5 + 11 * storm + rng.normal(0, 0.8, n)
        wdir10 = (220 + 35 * storm + rng.normal(0, 6, n)) % 360
        prcp = np.clip(0.1 + 3.2 * storm + rng.normal(0, 0.25, n), 0, None)
        mslp = 101600 - 2600 * storm + rng.normal(0, 170, n)

        rows.append(pd.DataFrame({
            "cell_id": grid_df["cell_id"].values,
            "valid_time": t,
            "gust10": gust10,
            "wspd10": wspd10,
            "wdir10": wdir10,
            "prcp_conv": prcp,
            "mslp": mslp,
        }))

    return pd.concat(rows, ignore_index=True)