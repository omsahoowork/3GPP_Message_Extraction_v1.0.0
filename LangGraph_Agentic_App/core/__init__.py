# core/__init__.py
"""Public API for the core package.

Import from here rather than from sub-modules directly so that callers are
insulated from internal reorganisation.

All heavy objects stay lazily loaded — importing this package should not pull
in optional vectorstore or embedding dependencies until a specific function is
actually used.
"""


def get_embeddings(*args, **kwargs):
    from core.embeddings import get_embeddings as impl

    return impl(*args, **kwargs)


def get_vectorstore(*args, **kwargs):
    from core.vectorstore import get_vectorstore as impl

    return impl(*args, **kwargs)


def get_vectorstore_508(*args, **kwargs):
    from core.vectorstore_508 import get_vectorstore_508 as impl

    return impl(*args, **kwargs)


def hybrid_search(*args, **kwargs):
    from core.retrieval import hybrid_search as impl

    return impl(*args, **kwargs)


def rerank(*args, **kwargs):
    from core.retrieval import rerank as impl

    return impl(*args, **kwargs)


def retrieve(*args, **kwargs):
    from core.retrieval import retrieve as impl

    return impl(*args, **kwargs)


def build_rag_context(*args, **kwargs):
    from core.retrieval import build_rag_context as impl

    return impl(*args, **kwargs)


def retrieve_508_context(*args, **kwargs):
    from core.retrieval_508 import retrieve_508_context as impl

    return impl(*args, **kwargs)


def get_lte_loop_sequence(*args, **kwargs):
    from core.retrieval_508 import get_lte_loop_sequence as impl

    return impl(*args, **kwargs)


def get_nr_loop_sequence(*args, **kwargs):
    from core.retrieval_508 import get_nr_loop_sequence as impl

    return impl(*args, **kwargs)


from core.prompts import (
    CONTEXT_JSON_EXTRACTION_PROMPT,
    RAG_EXTRACTION_PROMPT,
    LLM_ONLY_PROMPT,
)

__all__ = [
    # singletons
    "get_embeddings",
    "get_vectorstore",
    "get_vectorstore_508",

    # retrieval
    "hybrid_search",
    "rerank",
    "retrieve",
    "build_rag_context",
    "retrieve_508_context",
    "get_lte_loop_sequence",
    "get_nr_loop_sequence",
    
    # prompts
    "CONTEXT_JSON_EXTRACTION_PROMPT",
    "RAG_EXTRACTION_PROMPT",
    "LLM_ONLY_PROMPT",
]
