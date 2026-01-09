# db/migrations/007_create_master_grid_hourly.py

import os
from bq_client import ensure_dataset_exists, dataset_ref, get_bq_client
from schema_utils import build_create_table_sql

def create_master_grid_hourly_table():
    """
    Creates:
      gold_master_grid_hourly

    Purpose:
      Model-ready master table at (grid_cell_id, valid_time_utc) granularity,
      enriched with CAP + FMI + Forest grid features.

    Partition:
      DATE(valid_time_utc)
    Cluster:
      grid_cell_id
    """
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_MASTER", "gold_master_grid_hourly")
    full_table_id = f"{ds}.{table_name}"

    columns = [
        ("grid_cell_id", "STRING"),
        ("valid_time_utc", "TIMESTAMP"),
        ("forest_snapshot_date", "DATE"),

        # FMI (wide)
        ("wind_speed_10m_ms", "FLOAT64"),
        ("wind_gust_ms", "FLOAT64"),
        ("wind_dir_deg", "FLOAT64"),
        ("temp_c", "FLOAT64"),
        ("precip_mm", "FLOAT64"),
        ("snow_load_kgm2", "FLOAT64"),
        ("fmi_model_run_utc", "TIMESTAMP"),

        # CAP (reduced)
        ("cap_active", "BOOL"),
        ("cap_max_severity_score", "FLOAT64"),
        ("cap_event_types", "STRING"),  # store as JSON string for simplicity (see note below)

        # Forest grid features (slow dimension)
        ("spruce_share", "FLOAT64"),
        ("pine_share", "FLOAT64"),
        ("other_share", "FLOAT64"),
        ("meanheight_m_aw", "FLOAT64"),
        ("meandiameter_cm_aw", "FLOAT64"),
        ("basalarea_aw", "FLOAT64"),
        ("meanage_aw", "FLOAT64"),
        ("maintreespecies_mode", "STRING"),
        ("soiltype_mode", "STRING"),
        ("drainagestate_mode", "STRING"),
        ("developmentclass_mode", "STRING"),

        ("built_at_utc", "TIMESTAMP"),
    ]

    ddl = build_create_table_sql(
        full_table_id=full_table_id,
        columns=columns,
        partition_field="valid_time_utc",     # -> PARTITION BY DATE(valid_time_utc)
        cluster_fields=["grid_cell_id"],
    )

    print(f"[bq] creating table if missing: {full_table_id}")
    print(ddl)
    client.query(ddl).result()
    print("[bq] master table ready")

if __name__ == "__main__":
    create_master_grid_hourly_table()
