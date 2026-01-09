import os
from bq_client import dataset_ref
from run_sql import run_sql

def create_grid_table():
    ds = dataset_ref()
    sql = f"""
        -- GRID
        CREATE TABLE IF NOT EXISTS `{ds}.dim_grid_1km` (
          grid_cell_id STRING NOT NULL,
          geom GEOGRAPHY NOT NULL,
          centroid GEOGRAPHY,
          area_m2 FLOAT64,
          created_at_utc TIMESTAMP
        )
        CLUSTER BY grid_cell_id;
        """
    run_sql(sql)
    print("Grid table created")

if __name__ == '__main__':
    create_grid_table()