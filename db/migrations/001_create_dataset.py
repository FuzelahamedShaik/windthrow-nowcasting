import os
from google.cloud import bigquery
from bq_client import get_bq_client

def create_dataset_if_missing():
    client = get_bq_client()
    project = client.project
    dataset_id = os.getenv("BQ_DATASET", "windthrow")
    location = os.getenv("BQ_LOCATION", "EU")

    ds_ref = bigquery.Dataset("{}.{}".format(project, dataset_id))
    ds_ref.location = location

    try:
        client.get_dataset(ds_ref)
        print("Dataset already exists: {}.{}".format(project, dataset_id))
    except Exception:
        client.create_dataset(ds_ref, exists_ok=True)
        print("Created dataset: {}.{} (location={})".format(project, dataset_id, location))

if __name__ == '__main__':
    create_dataset_if_missing()