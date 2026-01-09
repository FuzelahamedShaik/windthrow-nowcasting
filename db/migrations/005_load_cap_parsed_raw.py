import os
from pathlib import Path

from bq_client import ensure_dataset_exists, dataset_ref
from schema_utils import load_parquet_file_to_bq

def main():
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_CAP_PARSED_RAW", "fact_cap_parsed_raw")
    full_table_id = f"{ds}.{table_name}"

    root = Path(os.getenv("CAP_PARSED_DIR", "./data/interim/cap_parsed")).resolve()
    if not root.exists():
        raise FileNotFoundError(f"CAP_PARSED_DIR not found: {root}")

    # Most setups have cap_alerts_latest.parquet + maybe historical parquets
    preferred = root / "cap_alerts_latest.parquet"
    if preferred.exists():
        files = [preferred]
        print(f"[cap] loading latest: {preferred}")
    else:
        files = sorted(root.rglob("*.parquet"))
        print(f"[cap] cap_alerts_latest.parquet not found, loading {len(files)} parquet files under {root}")

    for p in files:
        load_parquet_file_to_bq(
            parquet_path=str(p),
            full_table_id=full_table_id,
            sample_rows_env="CAP_LOAD_SAMPLE_ROWS",   # set to 10 for test
        )

if __name__ == "__main__":
    main()
