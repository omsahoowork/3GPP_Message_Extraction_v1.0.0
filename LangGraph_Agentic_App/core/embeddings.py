# core/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL, MODEL_CACHE_DIR

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy singleton for the HuggingFace embedding model.

    The model is loaded on first call only, so importing this module does not
    trigger any heavy initialisation at import time.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            cache_folder=MODEL_CACHE_DIR,
            show_progress=True,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
