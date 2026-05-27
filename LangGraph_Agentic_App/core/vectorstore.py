# core/vectorstore.py
from langchain_chroma import Chroma
from core.embeddings import get_embeddings
from config import VECTORSTORE_DIR, COLLECTION_NAME
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=VECTORSTORE_DIR,
        )
    return _vectorstore