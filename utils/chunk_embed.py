# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from typing import List

# def embed_and_store_chunks(documents: List[Document], persist_path: str):
#     # 1. Split into chunks (preserving metadata like page number)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     split_docs = text_splitter.split_documents(documents)

#     # 2. Embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # 3. Create and persist vector DB
#     vectordb = Chroma.from_documents(
#         documents=split_docs,
#         embedding=embeddings,
#         persist_directory=persist_path
#     )
#     vectordb.persist()

#     return vectordb

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # ✅ updated import
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import


def embed_and_store_chunks(text: str, persist_path: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )

    # No need to call vectorstore.persist() — it auto-persists
    return vectorstore
