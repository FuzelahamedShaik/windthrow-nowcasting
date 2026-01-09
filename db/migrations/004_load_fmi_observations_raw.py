# db/migrations/004_load_fmi_observations_raw.py

import os
from pathlib import Path

from bq_client import ensure_dataset_exists, dataset_ref
from schema_utils import load_parquet_file_to_bq

def main():
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_FMI_OBS_RAW", "fact_fmi_observations_raw")
    full_table_id = f"{ds}.{table_name}"

    # Root folder containing YYYY/MM/DD/obs_*.parquet
    root = Path(os.getenv("FMI_OBS_DIR", "./data/interim/fmi/observations")).resolve()
    if not root.exists():
        raise FileNotFoundError(f"FMI_OBS_DIR not found: {root}")

    files = sorted(root.rglob("*.parquet"))
    print(f"[fmi] found {len(files)} parquet files under {root}")

    # Optional: only load the newest N files
    newest_n = int(os.getenv("FMI_LOAD_NEWEST_N", "0"))
    if newest_n > 0:
        files = files[-newest_n:]
        print(f"[fmi] limiting to newest {newest_n} files")

    for p in files:
        load_parquet_file_to_bq(
            parquet_path=str(p),
            full_table_id=full_table_id,
            sample_rows_env="FMI_LOAD_SAMPLE_ROWS",   # set to 10 for test
        )

if __name__ == "__main__":
    main()
