# db/migrations/006_create_forest_grid_features.py

import os
from bq_client import ensure_dataset_exists, dataset_ref, get_bq_client
from schema_utils import build_create_table_sql

def create_forest_grid_features_table():
    """
    Creates:
      gold_forest_grid_features

    This table stores area-weighted forest stand attributes aggregated to grid cells.
    Partition: snapshot_date (DATE)
    Cluster: grid_cell_id
    """
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_FOREST_GRID", "gold_forest_grid_features")
    full_table_id = f"{ds}.{table_name}"

    columns = [
        ("grid_cell_id", "STRING"),
        ("snapshot_date", "DATE"),
        ("stands_ingested_at_utc", "TIMESTAMP"),

        ("covered_area_m2", "FLOAT64"),
        ("coverage_ratio", "FLOAT64"),

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

        ("created_at_utc", "TIMESTAMP"),
    ]

    ddl = build_create_table_sql(
        full_table_id=full_table_id,
        columns=columns,
        partition_field="snapshot_date",      # DATE partition (schema_utils handles DATE)
        cluster_fields=["grid_cell_id"],
    )

    print(f"[bq] creating table if missing: {full_table_id}")
    print(ddl)
    client.query(ddl).result()
    print("[bq] forest grid features table ready")

if __name__ == "__main__":
    create_forest_grid_features_table()
