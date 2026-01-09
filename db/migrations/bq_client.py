import os
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

def get_bq_client():
    project = os.getenv("GCP_PROJECT")
    if not project:
        raise Exception("GCP_PROJECT environment variable is not set")
    return bigquery.Client(project=project)

def dataset_ref():
    project = os.getenv("GCP_PROJECT")
    dataset = os.getenv("BQ_DATASET", "windthrow")
    return "{}.{}".format(project, dataset)

def ensure_dataset_exists():
    client = get_bq_client()
    ds_id = dataset_ref()
    location = os.getenv("BQ_LOCATION", "EU")

    try:
        client.dataset(ds_id)
    except Exception:
        ds = bigquery.Dataset(ds_id)
        ds.location = location
        client.create_dataset(ds, exists_ok=True)
        print("[bq] created dataset {} ({})".format(ds_id, location))