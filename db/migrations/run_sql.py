from bq_client import *

def run_sql(sql):
    client = get_bq_client()
    job = client.query(sql)
    job.result()
    print("OK")