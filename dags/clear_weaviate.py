"""
## Delete a collection in Weaviate

CAUTION: This DAG will delete a specified collection in your Weaviate instance.
Meant to be used during development to reset Weaviate.
Please use it with caution.
"""

from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from airflow.decorators import dag, task
import os

# Provider your Weaviate conn_id here.
WEAVIATE_CONN_ID = os.getenv("WEAVIATE_CONN_ID", "weaviate_default")
# Provide the collection name to delete the schema.
WEAVIATE_COLLECTION_TO_DELETE = "MY_SCHEMA_TO_DELETE"


@dag(
    dag_display_name="🧼 Delete a Schema in Weaviate",
    schedule=None,
    start_date=None,
    catchup=False,
    description="CAUTION! Will delete a collection in Weaviate!",
    tags=["helper"]
)
def clear_weaviate():

    @task(
        task_display_name=f"Delete {WEAVIATE_COLLECTION_TO_DELETE} in Weaviate",
    )
    def delete_all_weaviate_schemas(COLLECTION_TO_DELETE=None):
        WeaviateHook(WEAVIATE_CONN_ID).delete_collections(COLLECTION_TO_DELETE)

    delete_all_weaviate_schemas(COLLECTION_TO_DELETE=WEAVIATE_COLLECTION_TO_DELETE)


clear_weaviate()
