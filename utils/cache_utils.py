import hashlib
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CACHE_DIR = "cached"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def save_vectorstore(hash_key: str, vectorstore):
    # Do nothing – persist is already called in embed_and_store_chunks
    pass

def load_vectorstore(hash_key: str):
    persist_path = os.path.join(CACHE_DIR, hash_key)
    if os.path.exists(persist_path):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_path
        )
    return None
