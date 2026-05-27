# tools/retrieval_tool.py
"""RAG retrieval tool — retrieve top-k spec chunks for a plain-text query.

Wraps :func:`core.retrieval.retrieve` and :func:`core.retrieval.build_rag_context`
as a single LangGraph ``@tool`` that accepts a question and returns a structured
context dict ready to be fed into the extraction prompt.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from core.retrieval import retrieve, build_rag_context


@tool
def retrieve_rag_context(
    question: Annotated[str, "The 3GPP procedure question to retrieve context for"],
    top_k_search: Annotated[int, "Number of top relevant chunks to retrieve in the search phase"] = 3,
    top_k_rerank: Annotated[int, "Number of top relevant chunks to retrieve in the re-rank phase"] = 1,
    top_vector_wt: Annotated[float, "Weight for vector similarity score in hybrid ranking"] = 0.2,
    top_mmr_wt: Annotated[float, "Weight for MMR score in hybrid ranking"] = 0.3,
    top_bm25_wt: Annotated[float, "Weight for BM25 score in hybrid ranking"] = 0.5,
    spec_series_filter: Annotated[str, "Optional spec family filter: \"36\" for LTE, \"38\" for NR"] = "",
    
) -> dict:
    """Retrieve the top-N most relevant 3GPP specification chunks for *question*.

    Uses hybrid vector + BM25 search followed by cross-encoder re-ranking.

    Returns
    -------
    dict
        Keys: ``source_doc`` (list of spec references),
        ``context`` (concatenated chunk texts, numbered ``[CHUNK i]``).
    """
    chunks = retrieve(
        question,
        top_k_search=top_k_search,
        top_k_rerank=top_k_rerank,
        top_vector_wt=top_vector_wt,
        top_mmr_wt=top_mmr_wt,
        top_bm25_wt=top_bm25_wt,
        spec_series_filter=spec_series_filter.strip() or None,
    )
    return build_rag_context(chunks, top_k_rerank)

@tool
def combine_context(
    context: Annotated[list[str], "The RAG context dict containing 'source_doc' and 'context' lists"],
) -> str:
    """Combine multiple RAG context dicts into a single context dict.

    This is useful when you have multiple RAG context dicts and want to merge them
    into a single context dict for further processing.


    Returns
    -------
    str
        A single string that combines all the contexts from the input list.
    """

    combined_context = ""
    for ctx in context:
        combined_context += ctx + "\n\n"
    return combined_context

    
