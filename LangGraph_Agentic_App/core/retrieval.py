# core/retrieval.py
"""Shared retrieval logic: hybrid search, re-ranking, and result formatting.

All heavy objects (vectorstore, BM25 index, reranker) are initialised lazily
the first time they are needed.  Importing this module has zero side effects.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any
from langchain_core.documents import Document
import joblib

from config import CHUNKS_DIR

import numpy as np
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from core.vectorstore import get_vectorstore


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_reranker: Any = None



def _get_reranker() -> Any:
    """Load (or return cached) cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        from config import RERANKER_MODEL, MODEL_CACHE_DIR

        _reranker = CrossEncoder(RERANKER_MODEL, cache_folder=MODEL_CACHE_DIR)
    return _reranker


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 2,
    top_vector_wt: float = 0.2,
    top_mmr_wt: float = 0.3,
    top_bm25_wt: float = 0.5,
    spec_series_filter: str | None = None,
) -> list[tuple[str, str]]:
    """Combine vector similarity and BM25 scores, deduplicate by breadcrumb,
    and merge sub-chunks that share the same breadcrumb.

    Returns
    -------
    list of (breadcrumb, merged_text) tuples, sorted by combined score descending.
    """

    vectorstore = get_vectorstore()

    documents = joblib.load(f"{CHUNKS_DIR}/chunks.pkl")
    

    # Initialise retrievers with appropriate search parameters
    retriever_similarity = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
    retriever_BM25 = BM25Retriever.from_documents(documents, search_kwargs={"k": top_k})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_similarity, retriever_mmr, retriever_BM25], weights=[top_vector_wt, top_mmr_wt, top_bm25_wt]
    )

    ensemble_relevant_docs = ensemble_retriever.invoke(query)

    relevant_sources = set()

    for doc in ensemble_relevant_docs:
        breadcrumb = doc.metadata.get("breadcrumb")
        if breadcrumb and _matches_spec_series(breadcrumb, spec_series_filter):
            relevant_sources.add(breadcrumb)

    # STEP 2: group FULL corpus by breadcrumb
    source_to_chunks = defaultdict(list)

    for doc in documents:
        breadcrumb = doc.metadata.get("breadcrumb")
        if not breadcrumb:
            continue

        if breadcrumb in relevant_sources and _matches_spec_series(breadcrumb, spec_series_filter):
            source_to_chunks[breadcrumb].append(doc.page_content)

    # STEP 3: take top-K per source (dedup-safe ordering preserved)
    results = []

    for breadcrumb, chunks in source_to_chunks.items():
        seen = set()
        filtered = []

        for chunk in chunks:
            if chunk in seen:
                continue
            seen.add(chunk)
            filtered.append(chunk)

            
        merged_content = "\n\n".join(
            f"\n{c}"
            for i, c in enumerate(filtered)
        )

        results.append((breadcrumb, merged_content))

    return results


def _matches_spec_series(breadcrumb: str | None, spec_series_filter: str | None) -> bool:
    """Return True if breadcrumb belongs to target 3GPP series.

    Examples:
      spec_series_filter="38" -> matches 38.XXX / 38523 / 38xxx docs
      spec_series_filter="36" -> matches 36.XXX / 36523 / 36xxx docs
    """
    if not spec_series_filter:
        return True
    if not breadcrumb:
        return False
    series = spec_series_filter.strip()
    return str(breadcrumb).lstrip().startswith(series)



def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int = 3,
) -> list[tuple[str, str, float]]:
    """Re-rank candidates using the cross-encoder model.

    Parameters
    ----------
    candidates:
        List of (breadcrumb, context_text) tuples.

    Returns
    -------
    Top-k list of (breadcrumb, context_text), sorted descending.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    texts = [ctx for _, ctx in candidates]
    pairs = [[query, text] for text in texts]
    scores: list[float] = reranker.predict(pairs).tolist()

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        (bc, ctx)
        for (bc, ctx), new_score in ranked[:top_k]
    ]


def retrieve(
    query: str,
    top_k_search: int = 3,
    top_k_rerank: int = 1,
    top_vector_wt: float = 0.2,
    top_mmr_wt: float = 0.3,
    top_bm25_wt: float = 0.5,
    spec_series_filter: str | None = None,
    
) -> list[tuple[str, str]]:
    """Full retrieval pipeline: hybrid search → cross-encoder re-rank.

    Returns
    -------
    Top-k list of (breadcrumb, context_text).
    """
    candidates = hybrid_search(
        query,
        top_k=top_k_search,
        top_vector_wt=top_vector_wt,
        top_mmr_wt=top_mmr_wt,
        top_bm25_wt=top_bm25_wt,
        spec_series_filter=spec_series_filter,
    )

    return rerank(query, candidates, top_k=top_k_rerank)



def find_sibling_chunks(
    breadcrumb: str,
    spec_series_filter: str | None = None,
) -> list[tuple[str, str]]:
    """Find all chunks from sibling sections sharing the same parent breadcrumb.

    A sibling is any breadcrumb whose parent (all but the last ' > ' segment)
    equals the parent of *breadcrumb*.  The selected breadcrumb itself is
    included in the result.

    Returns
    -------
    list of (breadcrumb, merged_context) pairs, one per unique sibling breadcrumb,
    sorted by numeric section prefix then lexicographically.
    """
    if not breadcrumb:
        return []

    # Determine parent: everything before the last ' > ' segment.
    sep = " > "
    parts = [p.strip() for p in breadcrumb.split(">") if p.strip()]
    if len(parts) < 3:
        # Not deep enough to have siblings; return just self.
        pass
    parent_parts = parts[:-1]
    parent_prefix = sep.join(parent_parts)  # normalised parent string

    # Derive the numeric parent prefix from the selected breadcrumb's last segment.
    # e.g. "8.1.4.1.2 Intra NR handover …" → (8, 1, 4, 1)
    def _extract_numeric(segment: str) -> tuple[int, ...]:
        m = re.search(r"\b(\d+(?:\.\d+)*)\b", segment)
        if not m:
            return ()
        try:
            return tuple(int(p) for p in m.group(1).split("."))
        except ValueError:
            return ()

    selected_numeric = _extract_numeric(parts[-1])       # e.g. (8,1,4,1,2)
    parent_numeric   = selected_numeric[:-1] if selected_numeric else ()  # e.g. (8,1,4,1)

    documents = joblib.load(f"{CHUNKS_DIR}/chunks.pkl")

    # Group all chunk texts by breadcrumb for siblings only.
    source_to_chunks: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        bc = str(doc.metadata.get("breadcrumb") or "").strip()
        if not bc:
            continue
        if spec_series_filter and not _matches_spec_series(bc, spec_series_filter):
            continue
        # Normalise breadcrumb for comparison
        bc_parts = [p.strip() for p in bc.split(">") if p.strip()]
        if len(bc_parts) < 2:
            continue

        # Primary match: breadcrumb path starts with parent_parts (structural descent)
        structural_match = (
            bc_parts[:len(parent_parts)] == parent_parts
            and len(bc_parts) > len(parent_parts)
        )

        # Secondary match: numeric section number starts with parent_numeric.
        # Catches chunks stored under a shallower/different path hierarchy but
        # belonging to the same numeric subtree (e.g. "8 > 8.1.2" when parent
        # was determined as "8 > 8.1").
        numeric_match = False
        if parent_numeric:
            bc_numeric = _extract_numeric(bc_parts[-1])
            numeric_match = (
                len(bc_numeric) > len(parent_numeric)
                and bc_numeric[:len(parent_numeric)] == parent_numeric
            )

        if structural_match or numeric_match:
            source_to_chunks[sep.join(bc_parts)].append(doc.page_content)

    # Merge chunks per breadcrumb and deduplicate.
    results: list[tuple[str, str]] = []
    for bc, chunks in source_to_chunks.items():
        seen: set[str] = set()
        merged_parts: list[str] = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                merged_parts.append(chunk)
        results.append((bc, "\n\n".join(merged_parts)))

    # Sort by numeric section prefix of the final segment, then lexicographically.
    def _numeric_key(bc: str) -> tuple:
        segs = [s.strip() for s in bc.split(">") if s.strip()]
        last = segs[-1] if segs else ""
        m = re.search(r"\b(\d+(?:\.\d+)*)\b", last)
        if not m:
            return ((), bc)
        try:
            return (tuple(int(p) for p in m.group(1).split(".")), bc)
        except ValueError:
            return ((), bc)

    results.sort(key=lambda x: _numeric_key(x[0]))
    return results


def build_rag_context(
    top_k_chunks: list[tuple[str, str]],
    top_k: int = 3,
) -> dict:
    """Assemble the RAG input dict consumed by the extraction prompt.

    Parameters
    ----------
    top_k_chunks:
        Output of :func:`retrieve` — list of (breadcrumb, context).

    Returns
    -------
    Dict with keys: ``source_doc``, ``context``.
    """
    top_sections = top_k_chunks[:top_k]
    sources = [bc for bc, _ in top_sections]
    contexts = [ctx for _, ctx in top_sections]

    return {
        "source_doc": sources,
        "context": contexts,
    }

