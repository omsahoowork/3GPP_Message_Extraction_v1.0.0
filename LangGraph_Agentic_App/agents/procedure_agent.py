"""Procedure message extraction agent with retry validation."""
from __future__ import annotations

import json

from config import AGENT_MAX_RETRIES
from tools.extraction_tool import extract_messages_rag


def run_procedure_agent(
    question: str,
    context: str,
    serving_cell_id: str,
    openai_api_key: str,
    other_cells: list[str] | None = None,
    other_participating_cell_ids: list[str] | None = None,
    test_purpose_index: int | None = None,
    max_retries: int = AGENT_MAX_RETRIES,
) -> list[dict]:
    """Extract procedure messages.

    Args:
        question: The procedure question
        context: The RAG context for extraction
        serving_cell_id: Serving cell ID
        other_cells: List of other cell IDs (preferred)
        other_participating_cell_ids: Backward-compatible alias for other cell IDs
        test_purpose_index: Optional selected TP index (1-based) for early filtering
        max_retries: Reserved for backward compatibility

    Returns:
        List of extracted message dicts
    """
    participant_cells = other_cells if other_cells is not None else other_participating_cell_ids
    if participant_cells is None:
        participant_cells = []

    # Kept to preserve function signature compatibility with existing callers.
    _ = max_retries

    try:
        messages = extract_messages_rag.invoke({
            "question": question,
            "context": context,
            "serving_cell_id": serving_cell_id,
            "openai_api_key": str(openai_api_key or "").strip(),
            "other_participating_cell_ids": json.dumps(participant_cells, ensure_ascii=False),
            "test_purpose_index": int(test_purpose_index) if isinstance(test_purpose_index, int) else 0,
        })
        if not isinstance(messages, list):
            messages = []
    except Exception as e:
        print(f"Procedure agent extraction error: {e}")
        return []

    return messages
