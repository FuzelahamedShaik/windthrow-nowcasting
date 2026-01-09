# db/migrations/005_create_cap_warnings.py

import os
from pathlib import Path

from bq_client import ensure_dataset_exists, dataset_ref, get_bq_client
from schema_utils import infer_bq_columns_from_parquet, build_create_table_sql

DEFAULT_SAMPLE_PARQUET = "./data/interim/cap_parsed/cap_alerts_latest.parquet"

def create_cap_parsed_raw_table(sample_parquet_path: str):
    """
    Raw table mirroring cap_alerts_latest.parquet

    Columns (from your parquet):
      identifier, sent, sender, status, msg_type, scope,
      language_first, event_first, severity_first, urgency_first, certainty_first,
      effective_first, onset_first, expires_first,
      headline_all, description_all, instruction_all, area_desc_first,
      source_url, bbox_match, raw_path, polygons_json

    Partition: DATE(sent) if TIMESTAMP
    Cluster: identifier, event_first
    """
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_CAP_PARSED_RAW", "fact_cap_parsed_raw")
    full_table_id = f"{ds}.{table_name}"

    p = Path(sample_parquet_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Sample parquet not found: {sample_parquet_path}\n"
            f"Set CAP_SAMPLE_PARQUET env var or update DEFAULT_SAMPLE_PARQUET."
        )

    cols = infer_bq_columns_from_parquet(str(p))
    cols_dict = {n: t for n, t in cols}

    # Strong stability overrides (helps avoid weird inference drift)
    overrides = {
        "identifier": "STRING",
        "sender": "STRING",
        "status": "STRING",
        "msg_type": "STRING",
        "scope": "STRING",

        "language_first": "STRING",
        "event_first": "STRING",
        "severity_first": "STRING",
        "urgency_first": "STRING",
        "certainty_first": "STRING",

        "headline_all": "STRING",
        "description_all": "STRING",
        "instruction_all": "STRING",
        "area_desc_first": "STRING",

        "source_url": "STRING",
        "raw_path": "STRING",
        "polygons_json": "STRING",

        # common: boolean indicator
        "bbox_match": "BOOL",

        # times (if parquet already has TIMESTAMP this won’t hurt)
        "sent": "TIMESTAMP",
        "effective_first": "TIMESTAMP",
        "onset_first": "TIMESTAMP",
        "expires_first": "TIMESTAMP",
    }
    cols = [(n, overrides.get(n, t)) for n, t in cols]
    cols_dict = {n: t for n, t in cols}

    partition_field = "sent" if cols_dict.get("sent") in ("TIMESTAMP", "DATETIME") else None

    # Cluster only allowed types
    cluster_fields = []
    for cf in ["identifier", "event_first"]:
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
    print("[bq] cap parsed raw table ready")

    return full_table_id


def create_cap_grid_alerts_table():
    """
    Grid-time CAP facts (built later from parsed polygons -> grid cells).
    Matches the target you described earlier.
    """
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_CAP_GRID", "fact_cap_grid_alerts")
    full_table_id = f"{ds}.{table_name}"

    columns = [
        ("grid_cell_id", "STRING"),
        ("valid_time_utc", "TIMESTAMP"),
        ("cap_identifier", "STRING"),

        ("event_type", "STRING"),
        ("severity", "STRING"),
        ("urgency", "STRING"),
        ("certainty", "STRING"),

        ("severity_score", "FLOAT64"),
        ("cap_active", "BOOL"),

        ("ingested_at_utc", "TIMESTAMP"),
    ]

    ddl = build_create_table_sql(
        full_table_id=full_table_id,
        columns=columns,
        partition_field="valid_time_utc",
        cluster_fields=["grid_cell_id", "cap_identifier"],
    )

    print(f"[bq] creating table if missing: {full_table_id}")
    print(ddl)
    client.query(ddl).result()
    print("[bq] cap grid alerts table ready")

    return full_table_id


def main():
    sample = os.getenv("CAP_SAMPLE_PARQUET", DEFAULT_SAMPLE_PARQUET)
    create_cap_parsed_raw_table(sample)
    create_cap_grid_alerts_table()


if __name__ == "__main__":
    main()
