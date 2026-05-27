from langchain_chroma import Chroma

from core.embeddings import get_embeddings
from config import COLLECTION_NAME_508, VECTORSTORE_508_DIR

_vectorstore_508 = None


def get_vectorstore_508():
    global _vectorstore_508
    if _vectorstore_508 is None:
        _vectorstore_508 = Chroma(
            collection_name=COLLECTION_NAME_508,
            embedding_function=get_embeddings(),
            persist_directory=VECTORSTORE_508_DIR,
        )
    return _vectorstore_508