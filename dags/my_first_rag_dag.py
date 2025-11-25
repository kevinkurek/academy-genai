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
_WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME")

_CREATE_COLLECTION_TASK_ID = "create_collection"
_COLLECTION_ALREADY_EXISTS_TASK_ID = "collection_already_exists"

@dag(
    dag_display_name="📚 Ingest Knowledge Base",
    start_date=datetime(2025, 5, 1),
    schedule="@daily",
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    tags=["RAG"], # What are tags for? Organizing and categorizing DAGs
    default_args={
        # "retries": 3,
        "retry_delay": duration(minutes=5),
        "owner": "ML Engineering Team",
    },
    doc_md=__doc__, # Documentation for the DAG, comes from the module docstring
    description="Ingest knowledge into the vector database for RAG.",
)
def my_first_rag_dag():

    # @task(retries=0) # if you don't want branching
    @task.branch(retries=0) # to branch task after checking collection
    def check_collection(conn_id: str,
                         collection_name: str,
                         create_collection_task_id: str,
                         collection_already_exists_task_id: str
                         ) -> str:
        # t_log.info(my_string)

        """
        Check if the target collection exists in the Weaviate schema.

        Args:
            conn_id: The connection ID to use.
            collection_name: The name of the collection to check.
            create_collection_task_id: The task ID to create the collection if it does not exist.
            collection_already_exists_task_id: The task ID to proceed if the collection already exists.
        Returns:
            str: Task ID of the next task to execute.
        """

        # connect to Weaviate using the Airflow connection `conn_id`
        t_log.info(f"Using Weaviate connection ID: {conn_id}")
        hook = WeaviateHook(conn_id)
        
        # check if the collection exists in Weaviate
        collection = hook.get_conn().collections.exists(collection_name)

        if collection:
            t_log.info(f"Collection {collection_name} already exists.")
            # return "The collection already exists."
            return collection_already_exists_task_id
        else:
            t_log.info(f"Collection {collection_name} does not exist yet.")
            # return "The collection does not exist yet."
            return create_collection_task_id

    check_collection_obj = check_collection(conn_id=_WEAVIATE_CONN_ID,
                     collection_name=_WEAVIATE_COLLECTION_NAME,
                     create_collection_task_id=_CREATE_COLLECTION_TASK_ID,
                     collection_already_exists_task_id=_COLLECTION_ALREADY_EXISTS_TASK_ID
                     )
    
    @task
    def create_collection(conn_id: str,
                          collection_name: str) -> None:
        """
        Create a collection in the Weaviate Schema.
        Args:
            conn_id: The connection ID to use.
            collection_name: The name of the collection to create.
        """
        from weaviate.classes.config import Configure
        
        hook = WeaviateHook(conn_id)
        hook.create_collection(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai()
        )

    # create a collection object to be used in the DAG
    create_collection_obj = create_collection(conn_id=_WEAVIATE_CONN_ID,
                                               collection_name=_WEAVIATE_COLLECTION_NAME)
    

    # empty operator to proceed if collection already exists
    collection_already_exists = EmptyOperator(
        task_id=_COLLECTION_ALREADY_EXISTS_TASK_ID
    )

    # weaviate is ready for ingestion
    weaviate_ready = EmptyOperator(task_id="weaviate_ready", trigger_rule="none_failed")
    
    # chain the tasks together
    chain(
        check_collection_obj,
        [create_collection_obj, collection_already_exists],
        weaviate_ready)
    
    @task
    def fetch_ingestion_folders_local_paths(ingestion_folders_local_path: str) -> list[str]:
        """
        Fetch the ingestion folders local paths from a comma-separated string.

        Args:
            ingestion_folders_local_path: Comma-separated string of local paths.

        Returns:
            list[str]: List of local paths.
        """
        
        # get all the folders in the given location
        folders = os.listdir(ingestion_folders_local_path)
        t_log.info(f"Found folders: {folders}")

        # return the full path of the folders
        return [
            os.path.join(ingestion_folders_local_path, folder) for folder in folders
        ]

    # return a list of ingestion folder paths
    fetch_ingestion_folders_local_paths_obj = fetch_ingestion_folders_local_paths(
        ingestion_folders_local_path=_INGESTION_FOLDERS_LOCAL_PATHS
    )

    @task(map_index_template="{{ my_custom_map_index }}")
    def extract_document_text(ingestion_folder_local_path: str):
        """
        Extract information from markdown files in a folder.
        Args:
            folder_path (str): Path to the folder containing markdown files.
        Returns:
            pd.DataFrame: A list of dictionaries containing the extracted information.
        """

        # get all markdown files in the folder
        files = [
            f for f in os.listdir(ingestion_folder_local_path) if f.endswith(".md")
        ]

        # initialize lists to store titles and texts
        titles = []
        texts = []

        # iterate through the files and extract titles and texts
        for file in files:
            file_path = os.path.join(ingestion_folder_local_path, file)
            titles.append(file.split(".")[0])  # use filename without extension as title

            # read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        
        # create a DataFrame to hold the extracted data
        document_df = pd.DataFrame(
            {
                "folder_path": ingestion_folder_local_path,
                "title": titles,
                "text": texts,
            }
        )

        # log the number of records extracted
        t_log.info(f"Number of records: {document_df.shape[0]}")

        # get the current context and define a custom map index variable
        from airflow.operators.python import get_current_context

        context = get_current_context()
        context['my_custom_map_index'] = (
            f"Extracted files from: {ingestion_folder_local_path}."
        )

        return document_df
    
    # dynamic task mapping to extract document text from each ingestion folder
    # example: if fetch_ingestion_folders_local_paths_obj is a list of 3 folder paths,
    # extract_document_text_obj will be a list of 3 dataframes with 3 dynamic task instances
    extract_document_text_obj = extract_document_text.expand(
        ingestion_folder_local_path=fetch_ingestion_folders_local_paths_obj
    )

    @task(map_index_template="{{ my_custom_map_index }}")
    def chunk_text(df):
        """
        Chunk the text in the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing the text to chunk.
        Returns:
            pd.DataFrame: The DataFrame with the text chunked.
        """

        # original imports when langchain.text_splitter was in langchain
        # from langchain.text_splitter import RecursiveCharacterTextSplitter
        # from langchain.schema import Document

        # updated imports for langchain 1.0.0+ where text_splitter is a separate package
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document  # alternative core Document class

        splitter = RecursiveCharacterTextSplitter()

        df["chunks"] = df["text"].apply(
            lambda x: splitter.split_documents([Document(page_content=x)])
        )

        df = df.explode("chunks", ignore_index=True)
        df.dropna(subset=["chunks"], inplace=True)
        df["text"] = df["chunks"].apply(lambda x: x.page_content)
        df.drop(["chunks"], inplace=True, axis=1)
        df.reset_index(inplace=True, drop=True)

        # get the current context and define the custom map index variable
        from airflow.operators.python import get_current_context

        context = get_current_context()

        context["my_custom_map_index"] = (
            f"Chunked files from a df of length: {len(df)}."
        )

        return df
    
    # dyanmically map chunk_text over the extracted document text dataframes
    # example: if 3 folders are ingested, extract_document_text_obj will be a list of 3 dataframes with 3 dynamic task instances
    # and chunk_text_obj will be a list of 3 dataframes with chunked text
    chunk_text_obj = chunk_text.expand(df=extract_document_text_obj)

    ingest_data = WeaviateIngestOperator.partial(
        task_id="ingest_data",
        conn_id=_WEAVIATE_CONN_ID,
        collection_name=_WEAVIATE_COLLECTION_NAME,
        map_index_template="Ingested fils from: {{ task.input_data.to_dict()['folder_path'][0] }}."
    ).expand(input_data=chunk_text_obj)

    # chain the ingestion after both weaviate_ready and chunk_text_obj is ready
    chain(
        [weaviate_ready, chunk_text_obj],
        ingest_data
    )


my_first_rag_dag()