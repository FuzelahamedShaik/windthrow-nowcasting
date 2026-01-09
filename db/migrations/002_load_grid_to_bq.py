# db/migrations/002_load_grid_to_bq.py

import os
from google.cloud import bigquery
from bq_client import get_bq_client, ensure_dataset_exists, dataset_ref

GRID_PARQUET = "./data/interim/grid/grid_1km_north_ostrobothnia.parquet"

def load_grid_raw():
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_id = f"{ds}.dim_grid_1km_raw"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    with open(GRID_PARQUET, "rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)

    job.result()
    print(f"[bq] loaded -> {table_id}")

if __name__ == "__main__":
    load_grid_raw()
