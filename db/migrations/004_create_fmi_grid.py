# db/migrations/004_create_fmi_grid.py

import os
from pathlib import Path

from bq_client import ensure_dataset_exists, dataset_ref, get_bq_client
from schema_utils import infer_bq_columns_from_parquet, build_create_table_sql

DEFAULT_SAMPLE_PARQUET = "./data/interim/fmi/observations/2025/12/29/obs_20251229T091205Z.parquet"

def create_fmi_observations_raw_table(sample_parquet_path: str):
    """
    Creates a raw table that mirrors FMI observations parquet:
      obs_time_utc, station_name, fmisid, lat, lon, parameter, value, unit

    Partition: DATE(obs_time_utc)
    Cluster: fmisid, parameter
    """
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_FMI_OBS_RAW", "fact_fmi_observations_raw")
    full_table_id = f"{ds}.{table_name}"

    if not Path(sample_parquet_path).exists():
        raise FileNotFoundError(
            f"Sample parquet not found: {sample_parquet_path}\n"
            f"Set FMI_OBS_SAMPLE_PARQUET env var or update DEFAULT_SAMPLE_PARQUET."
        )

    cols = infer_bq_columns_from_parquet(sample_parquet_path)
    cols_dict = {n: t for n, t in cols}

    # Ensure stable expected types (optional but recommended)
    # Your parquet likely has these:
    # obs_time_utc: TIMESTAMP, lat/lon/value: FLOAT64, fmisid: INT64
    overrides = {
        "obs_time_utc": "TIMESTAMP",
        "station_name": "STRING",
        "fmisid": "INT64",
        "lat": "FLOAT64",
        "lon": "FLOAT64",
        "parameter": "STRING",
        "value": "FLOAT64",
        "unit": "STRING",
    }
    cols = [(n, overrides.get(n, t)) for n, t in cols]
    cols_dict = {n: t for n, t in cols}

    partition_field = "obs_time_utc" if cols_dict.get("obs_time_utc") in ("TIMESTAMP", "DATETIME") else None

    # Cluster only on allowed types (no FLOAT64)
    cluster_fields = []
    for cf in ["fmisid", "parameter"]:
        if cols_dict.get(cf) in ("STRING", "INT64", "DATE", "TIMESTAMP", "DATETIME", "BOOL"):
            cluster_fields.append(cf)

    ddl = build_create_table_sql(
        full_table_id=full_table_id,
        columns=cols,
        partition_field=partition_field,
        cluster_fields=cluster_fields or None
    )

    print(f"[bq] creating table if missing: {full_table_id}")
    print(ddl)
    client.query(ddl).result()
    print("[bq] fmi observations raw table ready")

    return full_table_id


def create_fmi_grid_weather_table():
    """
    Creates the grid/hourly "wide" table you will build from raw obs later.
    (This is the same target table we discussed earlier.)
    """
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_FMI_GRID", "fact_fmi_grid_weather")
    full_table_id = f"{ds}.{table_name}"

    columns = [
        ("grid_cell_id", "STRING"),
        ("valid_time_utc", "TIMESTAMP"),
        ("model_run_utc", "TIMESTAMP"),

        ("wind_speed_10m_ms", "FLOAT64"),
        ("wind_gust_ms", "FLOAT64"),
        ("wind_dir_deg", "FLOAT64"),
        ("temp_c", "FLOAT64"),
        ("precip_mm", "FLOAT64"),
        ("snow_load_kgm2", "FLOAT64"),

        ("ingested_at_utc", "TIMESTAMP"),
    ]

    ddl = build_create_table_sql(
        full_table_id=full_table_id,
        columns=columns,
        partition_field="valid_time_utc",
        cluster_fields=["grid_cell_id", "model_run_utc"],
    )

    print(f"[bq] creating table if missing: {full_table_id}")
    print(ddl)
    client.query(ddl).result()
    print("[bq] fmi grid weather table ready")

    return full_table_id


def main():
    sample = os.getenv("FMI_OBS_SAMPLE_PARQUET", DEFAULT_SAMPLE_PARQUET)
    create_fmi_observations_raw_table(sample)
    create_fmi_grid_weather_table()


if __name__ == "__main__":
    main()