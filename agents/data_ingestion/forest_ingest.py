# windthrow_nowcasting/agents/data_ingestion/forest_ingest.py

"""
python agents/data_ingestion/forest_ingest.py --bbox "18,55,35,75" --layer "v1:stand"
curl -s "https://avoin.metsakeskus.fi/rajapinnat/v1/ows?service=WFS&request=GetCapabilities" \
| grep -oP '(?<=<Name>)[^<]+' \
| grep -i stand | head -n 50
v1:application_stand_06_16
v1:application_stand_10_10
v1:application_stand_10_30
v1:application_stand_10_91
v1:application_stand_11_30
v1:application_stand_11_50
v1:application_stand_11_90
v1:application_stand_11_91
v1:application_stand_12_17
v1:application_stand_13_50
v1:availability_stand
v1:completiondeclaration_stand_06_16
v1:completiondeclaration_stand_10_10
v1:completiondeclaration_stand_10_30
v1:completiondeclaration_stand_10_91
v1:completiondeclaration_stand_11_30
v1:completiondeclaration_stand_11_50
v1:completiondeclaration_stand_11_90
v1:completiondeclaration_stand_11_91
v1:completiondeclaration_stand_12_17
v1:completiondeclaration_stand_13_30
v1:completiondeclaration_stand_13_50
v1:stand
"""

import argparse
import hashlib
import os
import urllib
from datetime import datetime, timezone
import geopandas as gpd

# -------------- Helpers ----------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_bbox(s: str) -> None:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'minLon, minLat, maxLon, maxLat'")
    return tuple(map(float, parts))

def bbox_hash(bbox) -> str:
    return hashlib.md5(",".join(map(str, bbox)).encode("utf-8")).hexdigest()[:10]

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def make_wfs_url(base: str, layer: str, bbox, srs="EPSG:4326") -> str:
    """
    Generic WFS GetFeature URL. Works for GeoServer-style endpoints.
    """
    minx, miny, maxx, maxy = bbox
    # bbox param format: minx,miny,maxx,maxy,CRS
    return (
        f"{base}?"
        f"service=WFS&version=2.0.0&request=GetFeature"
        f"&typeNames={layer}"
        f"&outputFormat=application/json"
        f"&srsName={srs}"
        f"&bbox={minx},{miny},{maxx},{maxy},{srs}"
    )

# ------------ Main ingest -------------

def ingest_stands(
        wfs_base: str,
        layer: str,
        bbox,
        keep_cols: list[str] | None,
):
    url = make_wfs_url(wfs_base, layer, bbox)
    print(f"[forest] WFS url: {url}")

    try:
        gdf = gpd.read_file(url)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("HTTPError: ", e.code)
        print(body[:3000])
        raise
    if gdf.empty:
        return gdf

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.set_crs("EPSG:4326")

    if keep_cols:
        cols = [c for c in keep_cols if c in gdf.columns]
        cols = cols + (["geometry"] if "geometry" in cols else [])
        gdf = gdf[cols]

    gdf["source"] = "metsakeskus_wfs"
    gdf["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()

    return gdf

def main():
    p = argparse.ArgumentParser("Forest stand ingest (Metsäkeskus WFS)")
    p.add_argument("--bbox", required=True, help="minLon,minLat,maxLon,maxLat in EPSG:4326")
    p.add_argument("--wfs-base", default="https://avoin.metsakeskus.fi/rajapinnat/v1/ows", help="WFS/OWS base URL")
    p.add_argument("--layer", required=True, help="Layer name e.g. 'v1:stand' (example; check GetCapabilities)")
    p.add_argument("--raw-dir", default="data/raw/forest_stands")
    p.add_argument("--out-dir", default="data/interim/forest_stands")
    p.add_argument("--keep-cols", default="", help="Comma-separated columns to keep (optional)")

    args = p.parse_args()
    bbox = parse_bbox(args.bbox)
    h = bbox_hash(bbox)

    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()] or None

    ensure_dir(args.raw_dir)
    ensure_dir(args.out_dir)

    gdf = ingest_stands(
        wfs_base=args.wfs_base,
        layer=args.layer,
        bbox=bbox,
        keep_cols=keep_cols,
    )

    print(f"[forest] rows={len(gdf)} bbox={bbox}")

    if gdf.empty:
        print(
            "[forest] No features returned. Try: (1) different layer name, (2) smaller bbox, (3) check WFS capabilities.")
        return

    # Save RAW geojson
    raw_path = os.path.join(args.raw_dir, f"stands_{h}_{utc_stamp()}.geojson")
    gdf.to_file(raw_path, driver="GeoJSON")
    print(f"[forest] saved raw -> {raw_path}")

    # Save INTERIM parquet (store geometry as WKT to keep it simple)
    df = gdf.copy()
    df["geometry_wkt"] = df.geometry.to_wkt()
    df = df.drop(columns=["geometry"])
    out_path = os.path.join(args.out_dir, f"stands_{h}_{utc_stamp()}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[forest] saved parquet -> {out_path}")

if __name__ == "__main__":
    main()