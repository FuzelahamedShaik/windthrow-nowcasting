# db/migrations/schema_utils.py

import os
import tempfile
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from google.cloud import bigquery

from bq_client import get_bq_client

def _bq_type_from_pa(field: pa.Field) -> str:
    t = field.type

    if pa.types.is_timestamp(t):
        return "TIMESTAMP"
    if pa.types.is_date32(t) or pa.types.is_date64(t):
        return "DATE"

    if (pa.types.is_int8(t) or pa.types.is_int16(t) or pa.types.is_int32(t) or pa.types.is_int64(t) or
        pa.types.is_uint8(t) or pa.types.is_uint16(t) or pa.types.is_uint32(t) or pa.types.is_uint64(t)):
        return "INT64"

    if pa.types.is_float16(t) or pa.types.is_float32(t) or pa.types.is_float64(t):
        return "FLOAT64"

    if pa.types.is_boolean(t):
        return "BOOL"

    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "STRING"

    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "BYTES"

    # complex types: store as STRING in raw
    if pa.types.is_struct(t) or pa.types.is_list(t) or pa.types.is_map(t):
        return "STRING"

    return "STRING"


TYPE_OVERRIDES = {
    "MEASUREMENTDATE": "TIMESTAMP",
    "TREESTANDDATE": "TIMESTAMP",
    "CREATIONTIME": "TIMESTAMP",
    "UPDATETIME": "TIMESTAMP",

    # Keep as raw string, parse later in a view/table
    "ingested_at_utc": "STRING",

    # codes stored as float due to NaNs -> store INT64 in BQ raw
    "MAINTREESPECIES": "INT64",
    "CUTTINGTYPE": "INT64",
    "CUTTINGPROPOSALYEAR": "INT64",
    "SILVICULTURETYPE": "INT64",
    "SILVICULTUREPROPOSALYEAR": "INT64",
    "DITCHINGYEAR": "INT64",

    "STANDNUMBEREXTENSION": "STRING",
}


def infer_bq_columns_from_parquet(parquet_path: str):
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow

    cols = []
    for f in schema:
        name = f.name
        bq_type = _bq_type_from_pa(f)
        bq_type = TYPE_OVERRIDES.get(name, bq_type)
        cols.append((name, bq_type))

    return cols


def build_create_table_sql(full_table_id: str, columns, partition_field=None, cluster_fields=None) -> str:
    col_map = {n: t for n, t in columns}
    col_lines = [f"  `{name}` {typ}" for name, typ in columns]
    ddl = f"CREATE TABLE IF NOT EXISTS `{full_table_id}` (\n" + ",\n".join(col_lines) + "\n)"

    if partition_field:
        ptype = col_map.get(partition_field)
        if ptype == "DATE":
            ddl += f"\nPARTITION BY `{partition_field}`"
        elif ptype in ("TIMESTAMP", "DATETIME"):
            ddl += f"\nPARTITION BY DATE(`{partition_field}`)"

    if cluster_fields:
        fields = ", ".join([f"`{c}`" for c in cluster_fields])
        ddl += f"\nCLUSTER BY {fields}"

    ddl += ";"
    return ddl



def get_bq_schema_map(full_table_id: str) -> Dict[str, str]:
    client = get_bq_client()
    t = client.get_table(full_table_id)
    return {f.name: f.field_type for f in t.schema}  # e.g. INTEGER/STRING/TIMESTAMP


def ensure_source_file_column(full_table_id: str):
    client = get_bq_client()
    client.query(f"""
      ALTER TABLE `{full_table_id}`
      ADD COLUMN IF NOT EXISTS __source_file STRING
    """).result()


def delete_by_source_file(full_table_id: str, source_file: str):
    client = get_bq_client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("sf", "STRING", source_file)]
    )
    client.query(
        f"DELETE FROM `{full_table_id}` WHERE __source_file = @sf",
        job_config=job_config
    ).result()


def _bq_to_arrow_type(bq_type: str) -> pa.DataType:
    bt = (bq_type or "").upper()
    if bt == "STRING":
        return pa.string()
    if bt in ("INTEGER", "INT64"):
        return pa.int64()
    if bt in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        return pa.float64()
    if bt in ("BOOL", "BOOLEAN"):
        return pa.bool_()
    if bt == "TIMESTAMP":
        return pa.timestamp("us", tz="UTC")
    if bt == "DATE":
        return pa.date32()
    # fallback for anything else
    return pa.string()


def _normalize_batch_to_bq(
    batch: pa.RecordBatch,
    schema_map: Dict[str, str],
    ordered_cols: List[str],
) -> pa.RecordBatch:
    arrays = []
    for col in ordered_cols:
        target_bq = schema_map[col]
        target_arrow = _bq_to_arrow_type(target_bq)

        if col in batch.schema.names:
            arr = batch.column(batch.schema.get_field_index(col))

            # float -> int64 (NaN -> null)
            if target_bq in ("INTEGER", "INT64") and pa.types.is_floating(arr.type):
                arr = pc.if_else(pc.is_finite(arr), arr, None)

            arr = pc.cast(arr, target_arrow, safe=False)
        else:
            arr = pa.nulls(batch.num_rows, type=target_arrow)

        arrays.append(arr)

    return pa.RecordBatch.from_arrays(arrays, ordered_cols)


def write_normalized_parquet(
    original_path: str,
    schema_map: Dict[str, str],
    source_file: str,
    max_rows: Optional[int] = None,
    batch_size: int = 20_000,
) -> str:
    """
    Stream-read parquet and write a temp parquet matching BQ schema exactly.
    Adds __source_file column.
    Works for large files.
    """
    pf = pq.ParquetFile(original_path)

    # ensure __source_file in schema
    if "__source_file" not in schema_map:
        schema_map = dict(schema_map)
        schema_map["__source_file"] = "STRING"

    ordered_cols = list(schema_map.keys())
    out_schema = pa.schema([pa.field(c, _bq_to_arrow_type(schema_map[c])) for c in ordered_cols])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = tmp.name
    tmp.close()

    writer = pq.ParquetWriter(tmp_path, out_schema)

    written = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            rb = batch

            # inject __source_file
            if "__source_file" not in rb.schema.names:
                sf_arr = pa.array([source_file] * rb.num_rows, type=pa.string())
                rb = rb.append_column("__source_file", sf_arr)

            nb = _normalize_batch_to_bq(rb, schema_map, ordered_cols)

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


def load_parquet_file_to_bq(
    parquet_path: str,
    full_table_id: str,
    sample_rows_env: str = "",
):
    """
    Idempotent load: DELETE by __source_file then append.
    Always normalizes parquet to match BQ table schema.
    """
    client = get_bq_client()

    ensure_source_file_column(full_table_id)

    schema_map = get_bq_schema_map(full_table_id)
    source_file = os.path.basename(parquet_path)

    # delete old rows for this file
    delete_by_source_file(full_table_id, source_file)

    sample_rows = int(os.getenv(sample_rows_env, "0")) if sample_rows_env else 0
    max_rows = sample_rows if sample_rows > 0 else None

    normalized_path = write_normalized_parquet(
        original_path=parquet_path,
        schema_map=schema_map,
        source_file=source_file,
        max_rows=max_rows,
    )

    try:
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        with open(normalized_path, "rb") as f:
            job = client.load_table_from_file(f, full_table_id, job_config=job_config)
        job.result()

        print(f"[bq] loaded {parquet_path} -> {full_table_id} (rows={'ALL' if max_rows is None else max_rows})")
    finally:
        try:
            os.remove(normalized_path)
        except OSError:
            pass