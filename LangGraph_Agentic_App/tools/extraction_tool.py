# tools/extraction_tool.py
"""Message-sequence extraction tools — one per pipeline variant.

Three ``@tool`` functions are provided so the LangGraph router can dispatch to
the appropriate one based on graph state:

  extract_messages_rag  — context-only extraction (RAG pipeline)
  extract_messages_kg   — KG + context extraction (KG-RAG pipeline)
  extract_messages_llm  — LLM-only extraction (baseline, no grounding)
    generate_context_fields_json  — fill a user-facing JSON template from context

These tools call the LLM and return structured outputs for downstream graph use.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from config import LLM_MODEL, CONTEXT_TEMPLATE_PATH
from core.prompts import (
    CONTEXT_JSON_EXTRACTION_PROMPT,
    QUERY_ENHANCEMENT_PROMPT,
    RAG_EXTRACTION_PROMPT,
)
from core.utils import (
    infer_message_layer,
    split_layer_message,
    normalise_cell_id,
    normalise_message_parameters,
)


def _get_llm() -> ChatOpenAI:
    """Return a shared LLM instance (lazy, not module-level)."""
    return ChatOpenAI(model=LLM_MODEL, temperature=0.1)
# def _get_llm() -> ChatOllama:
#     """Return a shared LLM instance (lazy, not module-level)."""
#     return ChatOllama(model=LLM_MODEL, base_url="https://api.ollama.com", temperature=0.1)

# def _get_llm():
#     """Return Anthropic Sonnet 4.6 client for optional extraction paths."""
#     return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1)


def _parse_response(raw: str) -> list[dict]:
    """Parse the LLM response into a list of message dicts, tolerating minor formatting issues.
    
    Expected format: [{"name": "MsgName", "direction": "GNB_TO_UE", "cell_id": "SCell1", "layer": "RRC", "tp": "-"}, ...]
    Falls back gracefully for legacy string list or plain string responses.
    """
    text = raw.strip()
    # Strip accidental markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            normalised: list[dict] = []
            for item in result:
                if isinstance(item, dict):
                    raw_name = str(item.get("name", ""))
                    layer, name = split_layer_message(raw_name, str(item.get("layer", "")))
                    normalised.append({
                        "name": name,
                        "direction": str(item.get("direction", "UNKNOWN")),
                        "cell_id": normalise_cell_id(item.get("cell_id", "")),
                        "layer": layer,
                        "tp": str(item.get("tp", "-")).strip() or "-",
                        "message_parameters": normalise_message_parameters(item.get("message_parameters", [])),
                    })
                elif isinstance(item, str) and item.strip():
                    # Legacy plain-string fallback
                    layer, name = split_layer_message(item)
                    normalised.append({"name": name, "direction": "GNB_TO_UE", "cell_id": "", "layer": layer, "tp": "", "message_parameters": []})
            return normalised
    except json.JSONDecodeError:
        pass
    # Last resort: wrap the raw string so caller always gets a list
    layer, name = split_layer_message(text)
    return [{"name": name, "direction": "GNB_TO_UE", "cell_id": "", "layer": layer, "tp": "", "message_parameters": []}]


def _parse_json_object(raw: str) -> dict:
    """Parse the LLM response into a JSON object, tolerating fenced output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("LLM did not return a JSON object")
    return result


def _load_context_template() -> dict:
    """Load the default user-facing context extraction template."""
    with CONTEXT_TEMPLATE_PATH.open(encoding="utf-8") as handle:
        template = json.load(handle)
    if not isinstance(template, dict):
        raise ValueError("context_enhancement_user.json must contain a JSON object")
    return template


def _normalise_context_field_value(value: object, field_name: str = "") -> object:
    """Normalize extracted values while preserving structured JSON where present.

    Important: do not flatten nested arrays/dicts into strings, because
    serving_cell_id_or_number and other_participating_cell_id_or_number_list
    can intentionally carry tuple-like JSON arrays.
    """
    if value is None:
        return ""

    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            normalized_item = _normalise_context_field_value(item, field_name=field_name)
            if normalized_item not in ("", None, [], {}):
                cleaned[str(key)] = normalized_item
        return cleaned

    if isinstance(value, list):
        cleaned_list: list[object] = []
        for item in value:
            normalized_item = _normalise_context_field_value(item, field_name=field_name)
            if normalized_item not in ("", None, [], {}):
                cleaned_list.append(normalized_item)
        return cleaned_list

    return str(value).strip()


def _tp_matches_selected_test_purpose(tp_value: object, test_purpose_index: int | None) -> bool:
    """Return True when a message row belongs to the selected test purpose."""
    if not isinstance(test_purpose_index, int) or test_purpose_index <= 0:
        return True

    text = str(tp_value or "").strip()
    matches = re.findall(r"\d+", text)
    if not matches:
        return False

    return str(test_purpose_index) in matches


def _apply_test_purpose_filter(messages: list[dict], test_purpose_index: int | None) -> list[dict]:
    """Trim procedure rows up to the last row tagged with selected TP."""
    if not isinstance(test_purpose_index, int) or test_purpose_index <= 0:
        return messages

    table_rows = [msg for msg in messages if isinstance(msg, dict)]
    last_matching_index = -1
    for idx, row in enumerate(table_rows):
        if _tp_matches_selected_test_purpose(row.get("tp", ""), test_purpose_index):
            last_matching_index = idx

    if last_matching_index >= 0:
        return table_rows[: last_matching_index + 1]

    # Fallback for sparse/partial TP annotations.
    return table_rows


# ---------------------------------------------------------------------------
# Tool: RAG extraction
# ---------------------------------------------------------------------------
@tool
def query_enhancement_rag(
    query_config: Annotated[str, "The parsed query configuration from the user-supplied JSON file"],
    context: Annotated[
        str,
        "Retrieved specification text (output of retrieve_rag_context['context'])",
    ],
) -> str:
    """Enhance the query for 3GPP specification retrieval using RAG context only.
    Returns
    -------
    str       The enhanced query string to be used for downstream retrieval and extraction nodes.
    """
    llm = _get_llm()
    prompt = QUERY_ENHANCEMENT_PROMPT.format(query_config=query_config, context=context)
    raw = llm.invoke(prompt).content
    return raw

@tool
def extract_messages_rag(
    question: Annotated[str, "The 3GPP procedure question"],
    context: Annotated[
        str,
        "Retrieved specification text (output of retrieve_rag_context['context'])",
    ],
    serving_cell_id: Annotated[
        str,
        "Serving cell id/number used as default when no explicit cell is mentioned per message step.",
    ] = "",
    other_participating_cell_ids: Annotated[
        str,
        "JSON array string of additional participating cell ids/numbers, e.g. [\"NCell2\", \"SCell3\"].",
    ] = "[]",
    test_purpose_index: Annotated[
        int,
        "1-based selected test-purpose index. 0 disables TP filtering.",
    ] = 0,
) -> list[dict]:
    """Extract the ordered 3GPP signalling message sequence using RAG context only.

    The LLM is strictly grounded — it may only emit messages that appear in
    the provided *context*.

    Returns
    -------
    list[dict]
        Ordered list of dicts with "name", "direction" ("GNB_TO_UE" or "UE_TO_GNB"), "cell_id", and "layer".
    """
    llm = _get_llm()
    prompt = RAG_EXTRACTION_PROMPT.format(
        question=question,
        context=context,
        serving_cell_id=serving_cell_id,
        other_participating_cell_ids=other_participating_cell_ids,
    )
    raw = llm.invoke(prompt).content
    parsed = _parse_response(raw)
    return _apply_test_purpose_filter(parsed, test_purpose_index)


@tool
def generate_context_fields_json(
    context: Annotated[
        str,
        "The retrieved or selected context string used to populate the user-facing JSON fields",
    ],
    schema_json: Annotated[
        str,
        "Optional JSON object string describing the target fields. If empty, the default context_enhancement_user.json template is used.",
    ] = "",
) -> dict:
    """Populate the context selection JSON fields from a context string.

    Returns a JSON object that preserves the schema keys and fills each field
    using only the provided context so it can be shown to the user for review
    and selection.
    
    Retries up to 3 times on JSON parse failure with self-correction prompt.
    """
    llm = _get_llm()
    schema = _load_context_template() if not schema_json.strip() else json.loads(schema_json)
    if not isinstance(schema, dict):
        raise ValueError("schema_json must decode to a JSON object")

    prompt = CONTEXT_JSON_EXTRACTION_PROMPT.format(
        schema_json=json.dumps(schema, indent=2),
        context=context,
    )
    
    # Retry loop for JSON parse failures
    max_retries = 3
    for attempt in range(max_retries):
        raw = llm.invoke(prompt).content
        try:
            extracted = _parse_json_object(raw)
            return {
                key: _normalise_context_field_value(extracted.get(key, ""), field_name=key)
                for key in schema
            }
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                # Last attempt — return empty dict to avoid crashing
                print(f"Context extraction JSON parse failed after {max_retries} attempts: {e}")
                return {key: "" for key in schema}
            
            # Retry with correction prompt
            correction_prompt = f"""\
The previous response was not valid JSON. Please fix and return ONLY a valid JSON object.

Required keys: {list(schema.keys())}

{prompt}

Return ONLY the corrected JSON object, nothing else.
"""
            prompt = correction_prompt

