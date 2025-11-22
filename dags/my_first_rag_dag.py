"""
## Simple RAG DAG to ingest knowledge data into a vector database
This DAG
- ingests text data from markdown files
- chunks the text
- ingests the chunks into a Weaviate vector database
"""

from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from airflow.providers.weaviate.operators.weaviate import WeaviateIngestOperator
from pendulum import datetime, duration # for scheduling
import os
import logging
import pandas as pd

t_log = logging.getLogger("airflow.task")

# Variables used in the DAG
_INGESTION_FOLDERS_LOCAL_PATHS = os.getenv("INGESTION_FOLDERS_LOCAL_PATHS")
_WEAVIATE_CONN_ID = os.getenv("WEAVIATE_CONN_ID")
_WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME")
_WEAVIATE_VECTORIZER = os.getenv("WEAVIATE_VECTORIZER")
_WEAVIATE_SCHEMA_PATH = os.getenv("WEAVIATE_SCHEMA_PATH")

_CREATE_CLASS_TASK_ID = "create_class"
_CLASS_ALREADY_EXISTS_TASK_ID = "class_already_exists"

@dag(
    dag_display_name="📚 Ingest Knowledge Base",
    start_date=datetime(2025, 5, 1),
    schedule="@daily",
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    tags=["RAG"], # What are tags for? Organizing and categorizing DAGs
    default_args={
        "retries": 3,
        "retry_delay": duration(minutes=5),
        "owner": "ML Engineering Team",
    },
    doc_md=__doc__, # Documentation for the DAG, comes from the module docstring
    description="Ingest knowledge into the vector database for RAG.",
)
def my_first_rag_dag():

    @task.branch(retries=4)
    def check_class(my_string):
        t_log.info(my_string)

    check_class(my_string="Airflow is great!")

my_first_rag_dag()