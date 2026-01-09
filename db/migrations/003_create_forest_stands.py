# db/migrations/003_create_forest_stands.py

import os
import time
import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from google.cloud import bigquery

from bq_client import get_bq_client, ensure_dataset_exists, dataset_ref
from schema_utils import infer_bq_columns_from_parquet, build_create_table_sql

MANIFEST_FILE = "./db/logs/forest_stands_manifest.json"

# ---------------------------
# manifest helpers
# ---------------------------
def _load_manifest() -> set[str]:
    p = Path(MANIFEST_FILE)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text()))


def _save_manifest(processed: set[str]):
    p = Path(MANIFEST_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(processed), indent=2))


def _wait_until_file_stable(path: Path, checks=5, sleep_s=1.0):
    last = -1
    stable = 0
    for _ in range(checks * 6):
        if not path.exists():
            time.sleep(sleep_s)
            continue
        size = path.stat().st_size
        if size == last and size > 0:
            stable += 1
            if stable >= checks:
                return
        else:
            stable = 0
            last = size
        time.sleep(sleep_s)


# ---------------------------
# BigQuery helpers
# ---------------------------
def get_bq_schema_map(full_table_id: str) -> dict[str, str]:
    client = get_bq_client()
    t = client.get_table(full_table_id)
    # BigQuery returns "INTEGER" not "INT64"
    return {f.name: f.field_type for f in t.schema}


def ensure_source_file_column(full_table_id: str):
    client = get_bq_client()
    client.query(f"""
      ALTER TABLE `{full_table_id}`
      ADD COLUMN IF NOT EXISTS __source_file STRING
    """).result()


# ---------------------------
# Arrow normalization to match BigQuery schema exactly
# ---------------------------
def _bq_to_arrow_type(bq_type: str) -> pa.DataType:
    bt = (bq_type or "").upper()
    if bt in ("STRING",):
        return pa.string()
    if bt in ("INTEGER", "INT64"):
        return pa.int64()
    if bt in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        return pa.float64()
    if bt in ("BOOL", "BOOLEAN"):
        return pa.bool_()
    if bt in ("TIMESTAMP",):
        return pa.timestamp("us", tz="UTC")
    if bt in ("DATE",):
        return pa.date32()
    # Fallback
    return pa.string()


def normalize_batch_to_bq(batch: pa.RecordBatch, schema_map: dict[str, str], ordered_cols: list[str]) -> pa.RecordBatch:
    """
    Output must contain exactly the BigQuery columns (ordered_cols), with Arrow types compatible with schema_map.
    - Missing cols -> nulls
    - Extra cols -> dropped
    - Type mismatches -> cast
    """
    arrays = []
    for col in ordered_cols:
        target_bq = schema_map[col]
        target_arrow = _bq_to_arrow_type(target_bq)

        if col in batch.schema.names:
            arr = batch.column(batch.schema.get_field_index(col))

            # Special: float -> int (keep nulls)
            if (target_bq in ("INTEGER", "INT64")) and pa.types.is_floating(arr.type):
                # NaN -> null
                arr = pc.if_else(pc.is_finite(arr), arr, None)

            # cast (safe=False to coerce)
            arr = pc.cast(arr, target_arrow, safe=False)
        else:
            arr = pa.nulls(batch.num_rows, type=target_arrow)

        arrays.append(arr)

    return pa.RecordBatch.from_arrays(arrays, ordered_cols)


def write_normalized_parquet(
    original_path: str,
    schema_map: dict[str, str],
    source_file: str,
    max_rows: int | None = None,
    batch_size: int = 10_000
) -> str:
    """
    Stream-read parquet and write a new parquet that matches BigQuery table schema exactly.
    Works for huge files.
    """
    pf = pq.ParquetFile(original_path)

    # Ensure __source_file is part of output columns
    ordered_cols = list(schema_map.keys())
    if "__source_file" not in schema_map:
        # should not happen if ensure_source_file_column ran
        ordered_cols.append("__source_file")
        schema_map["__source_file"] = "STRING"

    # Build output schema in Arrow
    out_fields = [pa.field(c, _bq_to_arrow_type(schema_map[c])) for c in ordered_cols]
    out_schema = pa.schema(out_fields)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = tmp.name
    tmp.close()

    writer = pq.ParquetWriter(tmp_path, out_schema)

    written = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            rb = batch  # RecordBatch
            # Add __source_file to the incoming batch (if missing)
            if "__source_file" not in rb.schema.names:
                sf_arr = pa.array([source_file] * rb.num_rows, type=pa.string())
                rb = rb.append_column("__source_file", sf_arr)

            # Normalize to exact schema
            nb = normalize_batch_to_bq(rb, schema_map, ordered_cols)

            # Apply max_rows cut (for testing)
            if max_rows is not None:
                remaining = max_rows - written
                if remaining <= 0:
                    break
                if nb.num_rows > remaining:
                    nb = nb.slice(0, remaining)

            writer.write_table(pa.Table.from_batches([nb]))
            written += nb.num_rows

            if max_rows is not None and written >= max_rows:
                break
    finally:
        writer.close()

    return tmp_path


# ---------------------------
# Table creation
# ---------------------------
def ensure_forest_table_from_file(parquet_path: str):
    client = get_bq_client()
    ensure_dataset_exists()

    ds = dataset_ref()
    table_name = os.getenv("BQ_TABLE_FOREST_RAW", "fact_forest_stand_raw")
    full_table_id = f"{ds}.{table_name}"

    cols = infer_bq_columns_from_parquet(parquet_path)

    # choose partition field (must be DATE/TIMESTAMP)
    cols_dict = {name: typ for name, typ in cols}
    partition_field = "CREATIONTIME" if cols_dict.get("CREATIONTIME") == "TIMESTAMP" else None

    # type-safe clustering only
    cluster_fields = []
    for cf in ["DEVELOPMENTCLASS", "SOILTYPE", "MAINTREESPECIES"]:
        if cols_dict.get(cf) in ("STRING", "INT64", "DATE", "TIMESTAMP", "DATETIME", "BOOL"):
            cluster_fields.append(cf)

    ddl = build_create_table_sql(
        full_table_id=full_table_id,
        columns=cols,
        partition_field=partition_field,
        cluster_fields=cluster_fields or None
    )

    print(f"[bq] creating table if missing: {full_table_id}")
    client.query(ddl).result()

    # ensure idempotency column exists
    ensure_source_file_column(full_table_id)

    print("[bq] table ready")
    return full_table_id


# ---------------------------
# Loader (always normalize)
# ---------------------------
def load_parquet_into_forest_table(parquet_path: str, full_table_id: str):
    client = get_bq_client()
    schema_map = get_bq_schema_map(full_table_id)

    source_file = os.path.basename(parquet_path)

    # Idempotent delete: remove any prior loads of same file
    client.query(f"""
      DELETE FROM `{full_table_id}`
      WHERE __source_file = @sf
    """, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("sf", "STRING", source_file)]
    )).result()

    sample_rows = int(os.getenv("FOREST_LOAD_SAMPLE_ROWS", "0"))
    max_rows = sample_rows if sample_rows > 0 else None

    print(f"[bq] normalizing parquet (rows={'ALL' if max_rows is None else max_rows}): {parquet_path}")
    normalized_path = write_normalized_parquet(
        original_path=parquet_path,
        schema_map=schema_map,
        source_file=source_file,
        max_rows=max_rows
    )

    try:
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )

        with open(normalized_path, "rb") as f:
            job = client.load_table_from_file(f, full_table_id, job_config=job_config)
        job.result()
        print(f"[bq] loaded -> {full_table_id} (source_file={source_file})")
    finally:
        try:
            os.remove(normalized_path)
        except OSError:
            pass


# ---------------------------
# Watcher plumbing
# ---------------------------
def process_file(path: Path, processed: set[str]):
    if path.suffix.lower() != ".parquet":
        return

    key = str(path.resolve())
    if key in processed:
        return

    _wait_until_file_stable(path)

    full_table_id = ensure_forest_table_from_file(str(path))
    load_parquet_into_forest_table(str(path), full_table_id)

    processed.add(key)
    _save_manifest(processed)


class ParquetCreateHandler(FileSystemEventHandler):
    def __init__(self, processed: set[str]):
        self.processed = processed

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        print(f"[watch] new file: {path}")
        process_file(path, self.processed)


def main():
    stands_dir = Path(os.getenv("FOREST_STANDS_DIR", "./data/interim/forest_stands")).resolve()
    stands_dir.mkdir(parents=True, exist_ok=True)

    processed = _load_manifest()
    print(f"[watch] watching: {stands_dir}")
    print(f"[watch] already processed: {len(processed)} files")

    existing = sorted(stands_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    print(f"[watch] found {len(existing)} files")

    for p in existing:
        if str(p.resolve()) not in processed:
            print(f"[boot] processing existing: {p.name}")
            process_file(p, processed)

    handler = ParquetCreateHandler(processed)
    observer = Observer()
    observer.schedule(handler, str(stands_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()