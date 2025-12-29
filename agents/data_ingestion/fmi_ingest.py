# windthrow_nowcasting/agents/fmi_ingest.py

"""
FMI Ingestion Agent using fmiopendata

Ingests:
  1) Observations (multipointcoverage) -> station-time records
  2) Forecast grid (harmonie surface grid) -> cell-time-variable records + optional lat/lon arrays

Saves:
  data/raw/fmi/observations/... (metadata.json)
  data/interim/fmi/observations/YYYY/MM/DD/obs_*.parquet (+csv)
  data/raw/fmi/forecast_grid/... (metadata.json)
  data/interim/fmi/forecast_grid/init=YYYYMMDDTHH/forecast_*.parquet (+csv)

Usage examples:
  # last 1 hour observations (Finland bbox)
  python agents/data_ingestion/fmi_ingest.py obs --hours 1 --bbox 18,55,35,75

  # forecast grid for today 00-18Z
  python agents/data_ingestion/fmi_ingest.py grid --start "2025-12-29T00:00:00Z" --end "2025-12-29T18:00:00Z" --bbox 18,55,35,75

  # grid: last 24 hours window relative to now (useful for development)
  python agents/data_ingestion/fmi_ingest.py grid --recent-hours 24 --bbox 18,55,35,75
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from fmiopendata.wfs import download_stored_query

#---------------------------
# Config paths
# --------------------------

@dataclass
class FmiPaths:
    raw_root: str = "data/raw/fmi"
    interim_root: str = "data/interim/fmi"

    def raw_obs_dir(self) -> str:
        return os.path.join(self.raw_root, "observations")

    def raw_grid_dir(self) -> str:
        return os.path.join(self.raw_root, "forecast_grids")

    def interim_obs_dir(self, t:dt.datetime) -> str:
        return os.path.join(
            self.interim_root,
            "observations",
            f"{t.year:04d}", f"{t.month:02d}", f"{t.day:02d}"
        )

    def interim_grid_dir(self, init_time: dt.datetime) -> str:
        tag = init_time.strftime("%Y%m%dT%H")
        return os.path.join(
            self.interim_root,
            "forecast_grid",
            f"init={tag}",
        )

def ensure_dir (path: str) -> None:
    os.makedirs(path, exist_ok=True)

def iso_z(t: dt.datetime) -> str:
    """
    Return ISO8601 string with 'Z' (UTC)
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    t = t.astimezone(dt.timezone.utc)
    return t.isoformat(timespec="seconds").replace("+00:00", "Z")

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

# -----------------------------
# Flatten observations
# -----------------------------

def flatten_observations_multipoint(obs) -> pd.DataFrame:
    """
    Flatten MultiPointCoverage obs into a row-per-station-per-time table.

    obs.data structure:
      obs.data[obstime][station_name][param] -> {"value": ..., "units": ...}
    obs.location_metadata[station_name] -> {"fmisid":..., "latitude":..., "longitude":...}
    """
    rows: List[Dict] = []

    for t, stations in obs.data.items():
        # t is datetime (naive in examples) -> treat as UTC
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)

        for station_name, params in stations.items():
            meta = obs.location_metadata.get(station_name, {}) or {}
            fmisid = meta.get("fmisid")
            lat = meta.get("latitude")
            lon = meta.get("longitude")

            for param_name, payload in params.items():
                if payload is None:
                    continue
                val = payload.get("value")
                unit = payload.get("units")

                # fmiopendata can use numpy scalars; cast to python types
                if isinstance(val, (np.generic,)):
                    val = val.item()

                rows.append({
                    "obs_time_utc": t.isoformat(),
                    "station_name": station_name,
                    "fmisid": fmisid,
                    "lat": lat,
                    "lon": lon,
                    "parameter": param_name,
                    "value": val,
                    "unit": unit,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["obs_time_utc"] = pd.to_datetime(df["obs_time_utc"], utc=True)
    return df


# -----------------------------
# Flatten forecast grid
# -----------------------------

def flatten_grid(grid_obj, keep_latlon: bool = False) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Flatten fmiopendata.grid.Grid into a long dataframe:

    grid_obj.data[valid_time][level][dataset_name] = {"data": np.array, "units": ...}

    Returns:
      df_long with columns:
        valid_time_utc, level, dataset, unit, i, j, value
      plus (latitudes, longitudes) arrays if keep_latlon=True
    """
    rows = []
    lat_arr = getattr(grid_obj, "latitudes", None)
    lon_arr = getattr(grid_obj, "longitudes", None)

    for valid_time, levels in grid_obj.data.items():
        vt = valid_time
        if vt.tzinfo is None:
            vt = vt.replace(tzinfo=dt.timezone.utc)

        for level, datasets in levels.items():
            for dset_name, payload in datasets.items():
                unit = payload.get("units")
                data = payload.get("data")
                if data is None:
                    continue

                # data is 2D array [y, x] (commonly)
                arr = np.asarray(data)
                # iterate efficiently by flatten
                flat = arr.ravel()
                # indices
                h, w = arr.shape
                ii, jj = np.meshgrid(np.arange(w), np.arange(h))
                ii = ii.ravel()
                jj = jj.ravel()

                for k in range(flat.shape[0]):
                    v = flat[k]
                    if isinstance(v, (np.generic,)):
                        v = v.item()
                    rows.append({
                        "valid_time_utc": vt.isoformat(),
                        "level": int(level),
                        "dataset": str(dset_name),
                        "unit": unit,
                        "i": int(ii[k]),
                        "j": int(jj[k]),
                        "value": v,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)

    if keep_latlon:
        return df, lat_arr, lon_arr
    return df, None, None

# -----------------------------
# Ingest: observations
# -----------------------------

def ingest_observations(bbox: str, start: dt.datetime, end: dt.datetime, paths: FmiPaths, timeseries: bool = False) -> pd.DataFrame:
    """
    bbox format: "minLon,minLat,maxLon,maxLat"
    """
    ensure_dir(paths.raw_obs_dir())

    args = [
        f"bbox={bbox}",
        f"starttime={iso_z(start)}",
        f"endtime={iso_z(end)}",
    ]
    if timeseries:
        args.append("timeseries=True")

    obs = download_stored_query("fmi::observations::weather::multipointcoverage", args=args)

    # Save raw metadata (not the whole object, but key info)
    meta = {
        "stored_query": "fmi::observations::weather::multipointcoverage",
        "bbox": bbox,
        "starttime": iso_z(start),
        "endtime": iso_z(end),
        "timeseries": timeseries,
        "ingested_at_utc": iso_z(utc_now()),
        "n_times": len(getattr(obs, "data", {}) or {}),
        "n_stations_sample": len(next(iter(obs.data.values())).keys()) if getattr(obs, "data", None) else 0,
    }
    meta_path = os.path.join(paths.raw_obs_dir(), f"obs_meta_{iso_z(utc_now()).replace(':','').replace('-','')}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    df = flatten_observations_multipoint(obs)

    # Save structured
    if df.empty:
        return df

    # Partition by end date (or start date) - choose end date bucket
    out_dir = paths.interim_obs_dir(end.astimezone(dt.timezone.utc))
    ensure_dir(out_dir)

    stamp = iso_z(utc_now()).replace(":", "").replace("-", "")
    out_parquet = os.path.join(out_dir, f"obs_{stamp}.parquet")
    out_csv = os.path.join(out_dir, f"obs_{stamp}.csv")

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    return df


# -----------------------------
# Ingest: forecast grid
# -----------------------------

def ingest_forecast_grid(bbox: str, start: dt.datetime, end: dt.datetime, paths: FmiPaths, keep_latlon: bool = False) -> pd.DataFrame:
    """
    Robust ingest for FMI HARMONIE surface grid.
    - Tries init times newest->oldest until one parses with valid times
    - Emits diagnostics so you can see what's going on
    """
    ensure_dir(paths.raw_grid_dir())

    args = [
        f"starttime={iso_z(start)}",
        f"endtime={iso_z(end)}",
        f"bbox={bbox}",
    ]

    model_data = download_stored_query("fmi::forecast::harmonie::surface::grid", args=args)

    # model_data.data keys are init times
    init_times = sorted(model_data.data.keys())
    print(f"[grid] init_times returned: {[t.isoformat() for t in init_times]}")

    if not init_times:
        print("[grid] No init_times returned. Try adjusting time window (e.g., 00-18Z) or bbox.")
        return pd.DataFrame()

    # Try init times from newest to oldest
    df_long_final = pd.DataFrame()
    chosen_init = None
    chosen_obj = None
    lat_arr = lon_arr = None

    for init_time in sorted(init_times, reverse=True):
        print("[grid] entering init_time loop...")
        grid_obj = model_data.data[init_time]
        print(f"[grid] trying init_time={init_time} url={getattr(grid_obj, 'url', None)}")

        try:
            # This is the critical step (requires eccodes)
            grid_obj.parse(delete=True)
        except Exception as e:
            print(f"[grid] parse failed for init_time={init_time}: {e}")
            continue

        # After parse, grid_obj.data should have valid_time keys
        if not getattr(grid_obj, "data", None):
            print(f"[grid] parsed init_time={init_time} but grid_obj.data is empty")
            continue

        valid_times = sorted(grid_obj.data.keys())
        print(f"[grid] valid_times count={len(valid_times)} first={valid_times[0]} last={valid_times[-1]}")

        # Print one sample of levels + dataset names
        try:
            earliest = valid_times[0]
            levels = sorted(grid_obj.data[earliest].keys())
            print(f"[grid] levels @ earliest valid_time: {levels}")
            for lvl in levels:
                dsets = list(grid_obj.data[earliest][lvl].keys())
                print(f"[grid] level {lvl} dataset sample: {dsets[:8]}")
        except Exception as e:
            print(f"[grid] warning: could not inspect datasets: {e}")

        # Flatten
        df_long, lat_arr, lon_arr = flatten_grid(grid_obj, keep_latlon=keep_latlon)

        print(f"[grid] flattened rows={len(df_long)}")
        if len(df_long) == 0:
            continue

        chosen_init = init_time
        chosen_obj = grid_obj
        df_long_final = df_long
        break

    if df_long_final.empty:
        print("[grid] Could not parse/flatten any init time. Most common cause: eccodes not installed.")
        return df_long_final

    # Save metadata
    meta = {
        "stored_query": "fmi::forecast::harmonie::surface::grid",
        "bbox": bbox,
        "starttime": iso_z(start),
        "endtime": iso_z(end),
        "selected_init_time": chosen_init.isoformat() if chosen_init else None,
        "available_init_times": [t.isoformat() for t in init_times],
        "ingested_at_utc": iso_z(utc_now()),
        "keep_latlon": keep_latlon,
    }
    meta_path = os.path.join(paths.raw_grid_dir(),
                             f"grid_meta_{iso_z(utc_now()).replace(':', '').replace('-', '')}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Save structured outputs partitioned by init time
    out_dir = paths.interim_grid_dir(
        chosen_init.replace(tzinfo=dt.timezone.utc) if chosen_init.tzinfo is None else chosen_init)
    ensure_dir(out_dir)

    stamp = iso_z(utc_now()).replace(":", "").replace("-", "")
    out_parquet = os.path.join(out_dir, f"forecast_{stamp}.parquet")
    out_csv = os.path.join(out_dir, f"forecast_{stamp}.csv")

    df_long_final.to_parquet(out_parquet, index=False)
    df_long_final.to_csv(out_csv, index=False)

    if keep_latlon and lat_arr is not None and lon_arr is not None:
        np.save(os.path.join(out_dir, "latitudes.npy"), lat_arr)
        np.save(os.path.join(out_dir, "longitudes.npy"), lon_arr)

    print(f"[grid] saved parquet: {out_parquet}")
    return df_long_final


# -----------------------------
# CLI
# -----------------------------

def parse_bbox(s: str) -> str:
    # accept "18,55,35,75"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'minLon,minLat,maxLon,maxLat'")
    return ",".join(parts)


def parse_iso(s: str) -> dt.datetime:
    # Accept "YYYY-MM-DDTHH:MM:SSZ" or without Z
    s = s.strip()
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def main():
    parser = argparse.ArgumentParser("FMI ingestion agent (obs + forecast grid)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_obs = sub.add_parser("obs", help="Ingest observations multipointcoverage")
    p_obs.add_argument("--bbox", type=parse_bbox, default="18,55,35,75")
    p_obs.add_argument("--hours", type=int, default=1, help="Lookback window in hours")
    p_obs.add_argument("--timeseries", action="store_true", help="Request timeseries=True from FMI")
    p_obs.add_argument("--raw-root", type=str, default="data/raw/fmi")
    p_obs.add_argument("--interim-root", type=str, default="data/interim/fmi")

    p_grid = sub.add_parser("grid", help="Ingest forecast grid (harmonie surface grid)")
    p_grid.add_argument("--bbox", type=parse_bbox, default="18,55,35,75")
    p_grid.add_argument("--start", type=str, default=None, help="ISO start (e.g. 2025-12-29T00:00:00Z)")
    p_grid.add_argument("--end", type=str, default=None, help="ISO end (e.g. 2025-12-29T18:00:00Z)")
    p_grid.add_argument("--recent-hours", type=int, default=None, help="If set, fetch (now-recent_hours .. now)")
    p_grid.add_argument("--keep-latlon", action="store_true", help="Save lat/lon arrays as .npy")
    p_grid.add_argument("--raw-root", type=str, default="data/raw/fmi")
    p_grid.add_argument("--interim-root", type=str, default="data/interim/fmi")

    args = parser.parse_args()
    paths = FmiPaths(raw_root=args.raw_root, interim_root=args.interim_root)

    if args.cmd == "obs":
        end = utc_now()
        start = end - dt.timedelta(hours=args.hours)
        df = ingest_observations(args.bbox, start, end, paths, timeseries=args.timeseries)
        print(f"[obs] rows={len(df)} bbox={args.bbox} start={iso_z(start)} end={iso_z(end)}")

    elif args.cmd == "grid":
        if args.recent_hours is not None:
            end = utc_now()
            start = end - dt.timedelta(hours=args.recent_hours)
        else:
            if not args.start or not args.end:
                raise SystemExit("grid requires either --recent-hours or both --start and --end")
            start = parse_iso(args.start)
            end = parse_iso(args.end)

        df = ingest_forecast_grid(args.bbox, start, end, paths, keep_latlon=args.keep_latlon)
        if df is None:
            print(f"[grid] rows=0 (df=None) bbox={args.bbox} start={iso_z(start)} end={iso_z(end)}")
        else:
            print(f"[grid] rows={len(df)} bbox={args.bbox} start={iso_z(start)} end={iso_z(end)}")

    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()