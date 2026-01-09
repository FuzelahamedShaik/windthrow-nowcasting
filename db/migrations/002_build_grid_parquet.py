# db/migrations/002_build_grid_parquet.py

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from datetime import datetime, timezone

# North Ostrobothnia (Pohjois-Pohjanmaa) approx bbox in EPSG:3067 (meters)
# If you want slightly wider coverage, increase margins by ~50-100km.
NORTH_OSTROBOTHNIA_BBOX_3067 = (
    250000,   # min easting
    7050000,  # min northing
    650000,   # max easting
    7550000   # max northing
)

CELL_SIZE_M = 1000
OUT_PARQUET = "./data/interim/grid/grid_1km_north_ostrobothnia.parquet"

def build_grid_1km_parquet(bbox_3067=NORTH_OSTROBOTHNIA_BBOX_3067,
                          cell_size=CELL_SIZE_M,
                          out_path=OUT_PARQUET):
    minx, miny, maxx, maxy = bbox_3067

    xs = np.arange(minx, maxx, cell_size, dtype=np.int64)
    ys = np.arange(miny, maxy, cell_size, dtype=np.int64)

    # store created_at as UTC-naive (BigQuery TIMESTAMP compatible)
    created_at_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    records = []
    for x in xs:
        for y in ys:
            poly = box(x, y, x + cell_size, y + cell_size)
            grid_cell_id = f"3067_{cell_size}_E{x}_N{y}"
            records.append((grid_cell_id, poly, x + cell_size / 2.0, y + cell_size / 2.0))

    gdf = gpd.GeoDataFrame(
        records,
        columns=["grid_cell_id", "geom_3067", "cx", "cy"],
        geometry="geom_3067",
        crs="EPSG:3067",
    )

    gdf["centroid_3067"] = gpd.points_from_xy(gdf["cx"], gdf["cy"], crs="EPSG:3067")
    gdf["area_m2"] = gdf["geom_3067"].area
    gdf["created_at_utc"] = created_at_utc

    # Transform polygons to WGS84 for WKT (BigQuery GEOGRAPHY expects lon/lat)
    gdf_poly_wgs = gdf.set_geometry("geom_3067").to_crs("EPSG:4326")
    gdf_cent_wgs = gpd.GeoSeries(gdf["centroid_3067"], crs="EPSG:3067").to_crs("EPSG:4326")

    out = pd.DataFrame({
        "grid_cell_id": gdf_poly_wgs["grid_cell_id"].astype(str),
        "geom_wkt": gdf_poly_wgs.geometry.to_wkt(),
        "centroid_wkt": gdf_cent_wgs.to_wkt(),
        "area_m2": gdf["area_m2"].astype(float),
        "created_at_utc": pd.to_datetime(created_at_utc),  # tz-naive
    })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[grid] bbox_3067={bbox_3067} cell_size={cell_size}m")
    print(f"[grid] wrote {len(out):,} cells -> {out_path}")

if __name__ == "__main__":
    build_grid_1km_parquet()