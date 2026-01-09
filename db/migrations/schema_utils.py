# db/migrations/schema_utils.py

import pyarrow as pa
import pyarrow.parquet as pq

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