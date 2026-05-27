from __future__ import annotations

from pathlib import Path

import joblib, re
from langchain_core.documents import Document

from config import CHUNKS_508_DIR

_documents_508: list[Document] | None = None


def _normalise_state_token(token: str) -> str:
    cleaned = re.sub(r"\bSTATE\b", "", str(token), flags=re.IGNORECASE)
    return "".join(cleaned.split()).upper()


def _parse_loop_path(loop_path: str | list[str]) -> list[str]:
    if isinstance(loop_path, list):
        return [_normalise_state_token(state) for state in loop_path if str(state).strip()]
    return [
        _normalise_state_token(state)
        for state in str(loop_path).split("->")
        if str(state).strip()
    ]


def _build_lte_transition_query(from_state: str, to_state: str) -> str:
    """Build a table-agnostic LTE transition lookup query."""
    return f"UE transition procedure from state {from_state} to state {to_state}"


def _build_nr_rrc_lookup_query(rrc_state: str) -> str:
    """Build a table-agnostic NR RRC-state lookup query."""
    return f"RRC procedure table for {rrc_state}"


def _get_documents_508() -> list[Document]:
    global _documents_508
    if _documents_508 is None:
        chunks_path = Path(CHUNKS_508_DIR) / "chunks.pkl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"508 chunks not found at {chunks_path}. Run chunking/chunking_508.py first."
            )
        _documents_508 = joblib.load(str(chunks_path))
    return _documents_508


def _find_documents(*, chunk_kind: str, **filters: str) -> list[Document]:
    results: list[Document] = []
    for doc in _get_documents_508():
        metadata = doc.metadata
        if metadata.get("chunk_kind") != chunk_kind:
            continue
        if any(str(metadata.get(key)) != str(value) for key, value in filters.items()):
            continue
        results.append(doc)
    return results


def get_lte_loop_sequence(loop_path: str | list[str]) -> dict:
    states = _parse_loop_path(loop_path)
    sequence_chunks: list[dict] = []
    missing_transitions: list[str] = []

    for from_state, to_state in zip(states, states[1:]):
        transition_query = _build_lte_transition_query(from_state, to_state)
        matches = _find_documents(
            chunk_kind="lte_transition_sequence",
            transition_from=from_state,
            transition_to=to_state,
        )
        if not matches:
            missing_transitions.append(f"{from_state}->{to_state}")
            continue
        doc = matches[0]
        sequence_chunks.append(
            {
                "transition": f"{from_state}->{to_state}",
                "transition_query": transition_query,
                "source_doc": doc.metadata.get("breadcrumb", ""),
                "table_caption": doc.metadata.get("table_caption", ""),
                "context": doc.page_content,
            }
        )

    return {
        "rat": "lte",
        "loop_path": states,
        "sequence_chunks": sequence_chunks,
        "combined_context": "\n\n".join(item["context"] for item in sequence_chunks),
        "missing_transitions": missing_transitions,
    }


def get_nr_loop_sequence(loop_path: str | list[str]) -> dict:
    states = _parse_loop_path(loop_path)
    state_chunks: list[dict] = []
    missing_states: list[str] = []

    for state_id in states:
        state_docs = _find_documents(chunk_kind="nr_state_definition", state_id=state_id)
        if not state_docs:
            missing_states.append(state_id)
            continue
        state_doc = state_docs[0]
        rrc_state = str(state_doc.metadata.get("rrc_state", ""))
        rrc_lookup_query = _build_nr_rrc_lookup_query(rrc_state)
        procedure_docs = _find_documents(chunk_kind="nr_rrc_sequence_table", rrc_state=rrc_state)
        procedure_doc = procedure_docs[0] if procedure_docs else None
        state_chunks.append(
            {
                "state_id": state_id,
                "rrc_state": rrc_state,
            "rrc_lookup_query": rrc_lookup_query,
                "state_definition": state_doc.page_content,
                "state_source_doc": state_doc.metadata.get("breadcrumb", ""),
                "sequence_source_doc": procedure_doc.metadata.get("breadcrumb", "") if procedure_doc else "",
                "table_caption": procedure_doc.metadata.get("table_caption", "") if procedure_doc else "",
                "context": procedure_doc.page_content if procedure_doc else "",
            }
        )

    return {
        "rat": "nr",
        "loop_path": states,
        "state_chunks": state_chunks,
        "combined_context": "\n\n".join(item["context"] for item in state_chunks if item["context"]),
        "missing_states": missing_states,
    }


def retrieve_508_context(rat: str, loop_path: str | list[str]) -> dict:
    rat_value = str(rat).strip().lower()
    if rat_value == "lte":
        return get_lte_loop_sequence(loop_path)
    if rat_value == "nr":
        return get_nr_loop_sequence(loop_path)
    raise ValueError("rat must be either 'lte' or 'nr'")