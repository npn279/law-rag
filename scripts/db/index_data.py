import sys
sys.path.append("")

import os
import dotenv
dotenv.load_dotenv()

import logging

import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import StorageContext, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding


PARENT_INDEX = os.getenv("PARENT_INDEX")
CHILD_INDEX = os.getenv("CHILD_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

def index_parent(documents=None):
    """
    Index data to parent index
    Input:
    - dir_or_files: path to directory or list of file names
    """

    text_splitter = TokenTextSplitter.from_defaults(
        chunk_size=1024,
        chunk_overlap=512
    )

    documents = text_splitter.get_nodes_from_documents(documents)

    vector_store = ElasticsearchStore(
        index_name=PARENT_INDEX, es_url="http://localhost:9200"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.embed_model = None

    index = VectorStoreIndex(
        documents,
        storage_context=storage_context,
    )
    return documents

def index_child(documents=None):
    """
    Index data to child index
    documents get from parent
    Input:
    - documents: list of documents
    """
    text_splitter = TokenTextSplitter.from_defaults(
        chunk_size=512,
        chunk_overlap=256
    )

    documents = index_parent(documents)
    documents = text_splitter.get_nodes_from_documents(documents)
    vector_store = ElasticsearchStore(
        index_name=CHILD_INDEX, es_url="http://localhost:9200"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.embed_model = OpenAIEmbedding(
        model=OPENAI_EMBEDDING_MODEL_NAME,
        dimensions=1536
    )

    index = VectorStoreIndex(
        documents,
        storage_context=storage_context,
    )
    

def index_data(dir_or_files=None):
    if isinstance(dir_or_files, str):
        documents = SimpleDirectoryReader(input_dir=dir_or_files).load_data()
    else:
        documents = SimpleDirectoryReader(input_files=dir_or_files).load_data()
    
    documents = index_parent(documents)
    index_child(documents)

if __name__ == "__main__":
    index_data('data/law_files')
