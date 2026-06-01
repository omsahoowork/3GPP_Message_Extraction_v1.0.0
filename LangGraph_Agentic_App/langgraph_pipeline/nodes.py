# langgraph_pipeline/nodes.py
"""Node functions for the LangGraph pipeline.

Every function in this file is a LangGraph *node*: it receives the current
``PipelineState`` dict, does exactly one unit of work, and returns a dict
containing only the state fields it wants to update.  LangGraph merges the
returned dict into the existing state automatically.

RULE OF THUMB
─────────────
- Return only the keys you changed.  Never return the full state.
- Catch exceptions locally and set ``{"error": "..."}`` so the router in
  graph.py can route to END gracefully instead of crashing the graph.

HOW TO ADD A NODE
─────────────────
1. Write a function ``my_node(state: PipelineState) -> dict`` here.
2. Register it in graph.py with ``graph.add_node("my_node", my_node)``.
3. Wire edges from/to it in graph.py.
4. Add any new state fields it produces to state.py.
"""
from __future__ import annotations
import csv
from functools import lru_cache
import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.types import interrupt
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

from langchain_core.output_parsers import JsonOutputParser

from langgraph_pipeline.state import PipelineState
from core.embeddings import get_embeddings
from tools.retrieval_tool import retrieve_rag_context, combine_context
from core.retrieval import find_sibling_chunks
from tools.retrieval_508_tool import retrieve_508_context
from tools.extraction_tool import (
    generate_context_fields_json,
    query_enhancement_rag,
    extract_messages_rag,
)
from core.utils import (
    infer_message_layer,
    split_layer_message,
    normalise_cell_id,
    normalise_message_parameters,
    normalise_message_sequence,
    assign_default_cell_id,
    get_cell_tuples,
    get_cell_metadata,
    filter_sib_system_messages,
    normalise_sib_state,
    infer_rat_label,
)
from agents import (
    run_sib_agent,
    run_ue_transition_agent,
    run_procedure_agent,
    run_shortlist_agent,
)
# Render JPG with matplotlib (no Mermaid .mmd intermediary file).
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from config import LLM_MODEL, QUERY_CONFIG_PATH, QUERY_ENHANCE_VECTOR_WEIGHT, QUERY_ENHANCE_MMR_WEIGHT, QUERY_ENHANCE_BM25_WEIGHT, ANSWER_EXTRACT_VECTOR_WEIGHT, ANSWER_EXTRACT_MMR_WEIGHT, ANSWER_EXTRACT_BM25_WEIGHT, QUERY_ENHANCE_TOP_SEARCH, QUERY_ENHANCE_TOP_RERANK, ANSWER_EXTRACT_TOP_SEARCH, ANSWER_EXTRACT_TOP_RERANK, VISUALIZATIONS_DIR, EVALUATION_DIR
from core.prompts import (
    LTE_TRANSITION_TABLE_MESSAGE_PROMPT,
    NR_RRC_TABLE_MESSAGE_PROMPT,
    SIB_MESSAGE_EXTRACTION_PROMPT,
    SIBLING_OPTION_SHORTLIST_PROMPT,
    LLM_ONLY_PROMPT,
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0.1)

# def _get_llm() -> ChatOllama:
#     """Return a shared LLM instance (lazy, not module-level)."""
#     return ChatOllama(model=LLM_MODEL, base_url="https://api.ollama.com", temperature=0.1)

# def _get_llm():
#     """Return Anthropic Sonnet 4.6 client for optional extraction paths."""
#     return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1)


def _get_json_parser() -> JsonOutputParser:
    """Return a JsonOutputParser for LLM output parsing."""
    return JsonOutputParser()


def _split_context_chunks(merged_str: str) -> list[str]:
    chunks = re.split(r"\[CHUNK \d+\]\n", str(merged_str or ""))
    return [c.strip() for c in chunks if c.strip()]


def _infer_message_layer(name: str) -> str:
    """Infer protocol layer from message name when not explicitly provided."""
    text = str(name or "").strip().upper()
    if not text:
        return "SYSTEM"
    if text in {"MIB"} or text.startswith("SIB") or "SYSTEM INFORMATION" in text:
        return "SYSTEM"
    for prefix, layer in (
        ("RRC", "RRC"),
        ("MAC", "MAC"),
        ("PDCP", "PDCP"),
        ("PHY", "PHY"),
        ("NAS", "NAS"),
        ("NGAP", "NGAP"),
        ("F1AP", "F1AP"),
        ("XNAP", "XNAP"),
        ("X2AP", "X2AP"),
        ("E1AP", "E1AP"),
        ("S1AP", "S1AP"),
        ("RLC", "RLC"),
        ("SDAP", "SDAP"),
        ("GTP", "GTP"),
        ("SCTP", "SCTP"),
    ):
        if text.startswith(prefix):
            return layer
    return "SYSTEM"


def _split_layer_message(raw_name: str, explicit_layer: str = "") -> tuple[str, str]:
    """Parse `layer: message` format and return `(layer, message_name)`.

    If no inline prefix exists, uses explicit_layer or inferred layer.
    """
    name_text = str(raw_name or "").strip()
    layer_text = str(explicit_layer or "").strip().upper()
    if ":" in name_text:
        parts = [part.strip() for part in name_text.split(":") if part.strip()]
        if len(parts) >= 2:
            candidate_layer = parts[0].upper()
            message_name = parts[-1]
            if message_name:
                return candidate_layer or _infer_message_layer(message_name), message_name
    return (layer_text or _infer_message_layer(name_text), name_text)


def _normalise_cell_id(cell_id: object) -> str:
    """Canonicalize cell identifiers so spacing differences do not create new lanes."""
    text = str(cell_id or "").strip()
    return re.sub(r"\s+", "", text)


def _normalise_message_parameters(value: object) -> list[str]:
    """Normalize message parameter collections into a de-duplicated string list."""
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    text = str(value or "").strip()
    if not text or text in {"-", "none", "None", "[]"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return _normalise_message_parameters(parsed)

    parts = [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
    dedup: list[str] = []
    for part in parts:
        if part not in dedup:
            dedup.append(part)
    return dedup


def _normalise_message_sequence(items: object, default_direction: str = "GNB_TO_UE") -> list[dict]:
    """Normalize mixed message outputs into [{name, direction, cell_id, layer}, ...]."""
    if not isinstance(items, list):
        return []

    normalised: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            direction = str(item.get("direction", default_direction)).strip() or default_direction
            cell_id = _normalise_cell_id(item.get("cell_id", ""))
            layer, clean_name = _split_layer_message(name, str(item.get("layer", "")))
            message_parameters = _normalise_message_parameters(item.get("message_parameters", []))
            normalised.append({
                "name": clean_name,
                "direction": direction,
                "cell_id": cell_id,
                "layer": layer,
                "message_parameters": message_parameters,
            })
        elif isinstance(item, str) and item.strip():
            layer, clean_name = _split_layer_message(item.strip())
            normalised.append({
                "name": clean_name,
                "direction": default_direction,
                "cell_id": "",
                "layer": layer,
                "message_parameters": [],
            })
    return normalised


def _coerce_cell_id_list(value: object) -> list[str]:
    """Coerce mixed scalar/list cell-id values into a clean list of strings."""
    if isinstance(value, list):
        return [_normalise_cell_id(v) for v in value if _normalise_cell_id(v)]

    text = _normalise_cell_id(value)
    if not text or text in {"-", "none", "None", "[]"}:
        return []

    # Try JSON list first.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [_normalise_cell_id(v) for v in parsed if _normalise_cell_id(v)]
        except Exception:
            pass

    # Fallback split for comma/pipe separated values.
    if "," in text or "|" in text:
        parts = re.split(r"[,|]", text)
        return [_normalise_cell_id(p) for p in parts if _normalise_cell_id(p)]

    return [text]


def _infer_rat_label(value: object, fallback: str = "") -> str:
    text = str(value or "").strip().upper()
    if "E-UTRA" in text or "LTE" in text:
        return "LTE"
    if re.search(r"\bNR\b", text):
        return "NR"

    fallback_text = str(fallback or "").strip().upper()
    if fallback_text in {"NR", "LTE"}:
        return fallback_text
    if fallback_text == "38":
        return "NR"
    if fallback_text == "36":
        return "LTE"
    return ""


def _normalise_sib_state(value: object) -> str:
    text = str(value or "").strip()
    if not text or text in {"-", "none", "None", "null", "[]"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _infer_rat_from_sib_state(sib_state: str, fallback_rat: str = "") -> str:
    text = str(sib_state or "").strip().upper()
    if not text:
        return _infer_rat_label("", fallback_rat)
    if "NR-" in text or re.search(r"\bNR\b", text):
        return "NR"
    if "SYSTEM INFORMATION COMBINATION" in text or re.search(r"\bE-UTRA\b|\bLTE\b", text):
        return "LTE"
    return _infer_rat_label(text, fallback_rat)


def _split_tuple_like_text(value: str) -> list[str]:
    text = str(value or "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def _coerce_cell_tuple(value: object, fallback_sib: str = "", fallback_rat: str = "") -> dict:
    """Normalize cell metadata into {cell_id, sib_state, rat}."""
    empty = {"cell_id": "", "sib_state": "", "rat": _infer_rat_label("", fallback_rat)}

    if isinstance(value, dict):
        cell_id = _normalise_cell_id(
            value.get("cell_id")
            or value.get("cell")
            or value.get("id")
            or value.get("name")
            or ""
        )
        sib_state = _normalise_sib_state(
            value.get("sib_state")
            or value.get("system_information_combination")
            or value.get("system_information_combinations")
            or fallback_sib
        )
        rat = _infer_rat_label(value.get("rat") or cell_id, fallback_rat)
        return {"cell_id": cell_id, "sib_state": sib_state, "rat": rat}

    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        parts = _split_tuple_like_text(str(value or ""))

    if not parts:
        return empty

    cell_id = _normalise_cell_id(parts[0]) if len(parts) >= 1 else ""
    sib_state = _normalise_sib_state(parts[1]) if len(parts) >= 2 else _normalise_sib_state(fallback_sib)
    rat_source = parts[2] if len(parts) >= 3 else cell_id
    rat = _infer_rat_label(rat_source or cell_id, fallback_rat)
    return {"cell_id": cell_id, "sib_state": sib_state, "rat": rat}


def _coerce_cell_tuple_list(value: object, fallback_sib: str = "", fallback_rat: str = "") -> list[dict]:
    if isinstance(value, list):
        if value and not any(isinstance(item, (list, dict)) for item in value):
            single = _coerce_cell_tuple(value, fallback_sib=fallback_sib, fallback_rat=fallback_rat)
            return [single] if single.get("cell_id") else []

        tuples: list[dict] = []
        for item in value:
            cell_tuple = _coerce_cell_tuple(item, fallback_sib=fallback_sib, fallback_rat=fallback_rat)
            if cell_tuple.get("cell_id"):
                tuples.append(cell_tuple)
        return tuples

    single = _coerce_cell_tuple(value, fallback_sib=fallback_sib, fallback_rat=fallback_rat)
    return [single] if single.get("cell_id") else []


def _get_default_rat_hint(context_json: dict, state: PipelineState | None = None) -> str:
    if isinstance(state, dict):
        spec_series = str(state.get("spec_series_filter") or "").strip()
        if spec_series == "38":
            return "NR"
        if spec_series == "36":
            return "LTE"

    context_blob = json.dumps(context_json or {}, ensure_ascii=False).upper()
    if "E-UTRA" in context_blob or " LTE" in context_blob:
        return "LTE"
    if "NR CELL" in context_blob or re.search(r"\bNR\b", context_blob):
        return "NR"
    return ""


def _get_cell_tuples(context_json: dict, state: PipelineState | None = None) -> tuple[dict, list[dict]]:
    """Resolve serving and participating cells as {cell_id, sib_state, rat}."""
    if not isinstance(context_json, dict):
        return {"cell_id": "SCell1", "sib_state": "", "rat": ""}, []

    fallback_rat = _get_default_rat_hint(context_json, state)

    serving = _coerce_cell_tuple(
        context_json.get("serving_cell_id_or_number", ""),
        fallback_sib="",
        fallback_rat=fallback_rat,
    )
    if serving.get("cell_id", "").lower() in {"", "none", "null", "-"}:
        serving = {"cell_id": "SCell1", "sib_state": serving.get("sib_state", ""), "rat": serving.get("rat", fallback_rat)}

    others = _coerce_cell_tuple_list(
        context_json.get("other_participating_cell_id_or_number_list", []),
        fallback_sib="",
        fallback_rat=fallback_rat,
    )

    serving_id = serving.get("cell_id", "")
    if serving_id:
        others = [cell for cell in others if cell.get("cell_id") != serving_id]

    global_sib_state = _normalise_sib_state(context_json.get("system_information_combinations", ""))
    if global_sib_state:
        global_sib_rat = _infer_rat_from_sib_state(global_sib_state, fallback_rat)
        all_cells = [serving, *others]

        # If RAT is still ambiguous, infer from the known cell RATs.
        if not global_sib_rat:
            rat_set = {
                str(cell.get("rat", "")).upper()
                for cell in all_cells
                if str(cell.get("rat", "")).upper() in {"NR", "LTE"}
            }
            if len(rat_set) == 1:
                global_sib_rat = next(iter(rat_set))

        if global_sib_rat in {"NR", "LTE"}:
            for cell in all_cells:
                if not _normalise_sib_state(cell.get("sib_state", "")) and str(cell.get("rat", "")).upper() == global_sib_rat:
                    cell["sib_state"] = global_sib_state

    return serving, others


def _get_cell_metadata(context_json: dict) -> tuple[str, list[str]]:
    """Resolve serving and participating cell IDs from selected context JSON."""
    serving, others = _get_cell_tuples(context_json)
    return serving.get("cell_id", "SCell1") or "SCell1", [cell.get("cell_id", "") for cell in others if cell.get("cell_id")]


def _assign_default_cell_id(
    messages: object,
    serving_cell_id: str,
    *,
    override_existing: bool = False,
) -> list[dict]:
    """Ensure each message dict contains cell_id with serving-cell fallback."""
    def _resolve_cell_id(value: object, fallback: str) -> str:
        text = _normalise_cell_id(value)
        if not text:
            return fallback

        lowered = re.sub(r"[^a-z0-9]", "", str(text).lower())
        if lowered in {"", "none", "null", "na", "unknown", "servingcellid", "servingcell", "defaultcellid"}:
            return fallback
        if "servingcell" in lowered:
            return fallback

        return text

    default_cell = _normalise_cell_id(serving_cell_id)
    normalised = _normalise_message_sequence(messages)
    updated: list[dict] = []
    for msg in normalised:
        current = _resolve_cell_id(msg.get("cell_id", ""), default_cell)
        layer = str(msg.get("layer", "")).strip().upper() or _infer_message_layer(str(msg.get("name", "")))
        updated.append({
            "name": msg.get("name", ""),
            "direction": msg.get("direction", "GNB_TO_UE"),
            "cell_id": default_cell if override_existing else (current or default_cell),
            "layer": layer,
            "message_parameters": _normalise_message_parameters(msg.get("message_parameters", [])),
        })
    return updated


def _message_sequence_to_text(items: object) -> str:
    seq = _normalise_message_sequence(items)
    return " ".join(message["name"] for message in seq)


def _coerce_messages_payload(value: object) -> list[dict]:
    """Coerce mixed agent/tool payloads into a message dict list."""
    def _strip_code_fence(text: str) -> str:
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))
        return text.strip()

    def _is_message_dict(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        return bool(str(item.get("name", "")).strip())

    def _extract(value_obj: object, depth: int = 0) -> list[dict]:
        # Keep recursion bounded for pathological payloads.
        if depth > 6:
            return []

        if isinstance(value_obj, list):
            messages: list[dict] = []
            for item in value_obj:
                if _is_message_dict(item):
                    messages.append(item)
                    continue

                # Common LLM content-block envelope:
                # [{"type": "text", "text": "[...]"}]
                if isinstance(item, dict) and "text" in item:
                    nested = _extract(item.get("text"), depth + 1)
                    if nested:
                        messages.extend(nested)
                        continue

                if isinstance(item, (dict, list, str)):
                    nested = _extract(item, depth + 1)
                    if nested:
                        messages.extend(nested)
            return messages

        if isinstance(value_obj, dict):
            if _is_message_dict(value_obj):
                return [value_obj]

            # Tool/agent wrapper keys.
            for key in ("messages", "message_sequence", "transition_message_sequence", "sib_message_sequence"):
                if key in value_obj:
                    nested = _extract(value_obj.get(key), depth + 1)
                    if nested:
                        return nested

            # Content-block dict: {"type": "text", "text": "..."}
            if "text" in value_obj:
                nested = _extract(value_obj.get("text"), depth + 1)
                if nested:
                    return nested

            return []

        if isinstance(value_obj, str):
            text = _strip_code_fence(value_obj)
            if not text:
                return []

            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError, TypeError):
                return []

            # Handle double-encoded JSON strings.
            if isinstance(parsed, str):
                return _extract(parsed, depth + 1)
            return _extract(parsed, depth + 1)

        return []

    return _extract(value)


_SIB_SYSTEM_INFO_PATTERN = re.compile(
    r"^SIB\d*$|SYSTEM\s+INFORMATION",
    flags=re.IGNORECASE,
)


def _filter_sib_system_messages(messages: list[dict]) -> list[dict]:
    """Remove SIB* and *SYSTEM INFORMATION* messages from a message list.

    These are broadcast/system-information messages that belong exclusively to
    the SYSTEM INFORMATION COMBINATION category and must not appear inside
    procedure or UE-state-transition sequences.
    """
    filtered: list[dict] = []
    for msg in messages:
        name = str(msg.get("name", "")).strip()
        if _SIB_SYSTEM_INFO_PATTERN.search(name):
            continue
        filtered.append(msg)
    return filtered


def _parse_llm_only_response(raw_text: str) -> list[dict]:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))
        text = text.strip()

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return _normalise_message_sequence(parsed, default_direction="UNKNOWN")
    return []


@lru_cache(maxsize=1)
def _get_ragas_embeddings():
    # Reuse the project embedding configuration (EMBED_MODEL) from config.py.
    return get_embeddings()


@lru_cache(maxsize=1)
def _get_ragas_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0.1)
    # return ChatOpenAI(model=LLM_MODEL, temperature=0.1)



def _load_sib_lookup_table(rat: str) -> str:
    """Load RAT-specific SIB lookup table from data_sib/<rat>_sib.md."""
    rat_value = str(rat).strip().lower()
    sib_filename = "nr_sib.md" if rat_value == "nr" else "lte_sib.md"
    sib_file = Path(__file__).parent.parent / "data_sib" / sib_filename
    if sib_file.exists():
        return sib_file.read_text(encoding="utf-8")
    return ""


def _normalise_system_info_combination(rat: str, value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().upper()
    if not text or text == "-":
        return ""

    if rat == "lte":
        match = re.search(r"SYSTEM INFORMATION COMBINATION\s*-?\s*([0-9]+[A-Z]?)", text)
        if match:
            return match.group(1)
        match = re.search(r"\b([0-9]+[A-Z]?)\b", text)
        if match:
            return match.group(1)
        return text

    match = re.search(r"\bNR\s*-?\s*([0-9]+[A-Z]?)\b", text)
    if match:
        return f"NR-{match.group(1)}"
    match = re.search(r"\b([0-9]+[A-Z]?)\b", text)
    if match:
        return f"NR-{match.group(1)}"
    return text


def _extract_sib_messages_from_sequence_text(sequence_text: str) -> list[str]:
    return re.findall(r"\bSIB\d+\b", str(sequence_text or ""), flags=re.IGNORECASE)


def _lookup_sib_messages_from_table(*, rat: str, system_info_combination: str, sib_lookup_table: str) -> list[str]:
    target = _normalise_system_info_combination(rat, system_info_combination)
    if not target:
        return []

    for raw_line in sib_lookup_table.splitlines():
        line = raw_line.strip()
        if not line.startswith("-") or ":" not in line:
            continue

        key_text, sequence_text = line[1:].split(":", 1)
        key = _normalise_system_info_combination(rat, key_text)
        if key != target:
            continue

        return [{"name": msg.upper(), "direction": "GNB_TO_UE", "layer": "SYSTEM"} for msg in _extract_sib_messages_from_sequence_text(sequence_text)]

    return []


def _resolve_rat_for_sib_search(
    *,
    state: PipelineState,
    system_info_combination: str,
) -> str:
    """Resolve which SIB lookup family to use (nr/lte)."""
    # Prefer explicit retrieval filter from state when available.
    # Accept both keys for backward compatibility.
    spec_filter = str(
        state.get("spec_specific_filter")
        or state.get("spec_series_filter")
        or ""
    ).strip()
    if spec_filter == "38":
        return "nr"
    if spec_filter == "36":
        return "lte"

    return _resolve_rat_for_state_sequence(state)


def _detect_spec_series_filter(query_config: dict) -> str | None:
    """Infer retrieval filter from user question/config.

    Returns:
        "38" for NR/5G, "36" for LTE/4G, or None if inconclusive.
    """
    rat = str(query_config.get("RAT", "")).upper()
    if "NR" in rat or "5G" in rat:
        return "38"
    if "LTE" in rat or "E-UTRA" in rat or "4G" in rat:
        return "36"

    blob = json.dumps(query_config, ensure_ascii=False).lower()
    has_nr = bool(
        re.search(r"\b(5g|nr|ng-ran|5gc|ts\s*38|38\.)\b", blob)
        or re.search(r"\bn\d{2,3}\b", blob)
    )
    has_lte = bool(re.search(r"\b(4g|lte|e-utra|eutra|epc|ts\s*36|36\.)\b", blob))

    if has_nr and not has_lte:
        return "38"
    if has_lte and not has_nr:
        return "36"
    return None


def _extract_sib_messages_for_combination(
    *,
    state: PipelineState,
    system_info_combination: str,
    rat_override: str | None = None,
) -> list[dict]:
    """Use LLM to extract SIB message sequence based on system information combination.
    
    Args:
        system_info_combination: The system information combination state from selected context.
    
    Returns:
        List of SIB message names in order, or empty list if not found.
    """
    if not system_info_combination or system_info_combination == "-":
        return []
    
    try:
        rat = str(rat_override or "").strip().lower() or _resolve_rat_for_sib_search(
            state=state,
            system_info_combination=system_info_combination,
        )
        sib_lookup_table = _load_sib_lookup_table(rat)
        if not sib_lookup_table:
            return []

        direct_match = _lookup_sib_messages_from_table(
            rat=rat,
            system_info_combination=system_info_combination,
            sib_lookup_table=sib_lookup_table,
        )
        if direct_match:
            return direct_match
        
        llm = _get_llm()
        parser = _get_json_parser()
        chain = llm | parser
        
        prompt = SIB_MESSAGE_EXTRACTION_PROMPT.format(
            system_info_combination=system_info_combination,
            sib_lookup_table=sib_lookup_table,
        )
        
        parsed = chain.invoke(prompt)
        sib_sequence = parsed.get("sib_message_sequence", [])
        
        if isinstance(sib_sequence, list):
            result: list[dict] = []
            for item in sib_sequence:
                if isinstance(item, dict) and "name" in item:
                    result.append({
                        "name": str(item["name"]),
                        "direction": "GNB_TO_UE",
                        "cell_id": str(item.get("cell_id", "")).strip(),
                        "layer": "SYSTEM",
                    })
                elif isinstance(item, str) and item.strip():
                    result.append({"name": item.upper(), "direction": "GNB_TO_UE", "cell_id": "", "layer": "SYSTEM"})
            return result
        return []
    except Exception as e:
        print(f"[_extract_sib_messages_for_combination] Failed to extract SIB messages: {e}")
        return []


def _normalise_state_key(state_value: str) -> str:
    cleaned = re.sub(r"[\*_`]", "", str(state_value), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    cleaned = re.sub(r"\bSTATE\b", "", cleaned).strip()
    return cleaned


def _load_state_loop_map(rat: str) -> dict[str, str]:
    states_file = "nr_states.md" if rat == "nr" else "lte_states.md"
    path = Path(__file__).parent.parent / "data_states" / states_file
    if not path.exists():
        return {}

    loop_map: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue

        state_cell = cells[0]
        # Loop path is always in the final column for both LTE and NR tables.
        # NR tables have extra middle columns (e.g., RRC state), so using a
        # fixed index would incorrectly capture "NR RRC_CONNECTED" etc.
        loop_path = cells[-1].replace("`", "").strip()
        if not state_cell or state_cell.lower() in {"state", "state id"}:
            continue
        if state_cell.startswith("---") or not loop_path:
            continue

        normalised = _normalise_state_key(state_cell)
        if normalised:
            loop_map[normalised] = loop_path

        # Keep a second key preserving an explicit "STATE X" form for LTE inputs.
        explicit = f"STATE {normalised}".strip()
        loop_map[explicit] = loop_path

    return loop_map


def _extract_state_key_from_ue_state(ue_state: str, rat: str) -> str:
    text = str(ue_state or "")
    if rat == "nr":
        match = re.search(r"\b\d[A-Z]-[A-Z]\b", text, flags=re.IGNORECASE)
        if match:
            return _normalise_state_key(match.group(0))
    else:
        match = re.search(r"\bSTATE\s*([0-9][A-Z]?(?:-[A-Z0-9]+)?)\b", text, flags=re.IGNORECASE)
        if match:
            return _normalise_state_key(match.group(1))

    return _normalise_state_key(text)


def _resolve_rat_for_state_sequence(state: PipelineState) -> str:
    series = str(state.get("spec_series_filter") or "").strip()
    if series == "38":
        return "nr"
    if series == "36":
        return "lte"

    rat_blob = json.dumps(state.get("query_config", {}), ensure_ascii=False).upper()
    if "NR" in rat_blob or "5G" in rat_blob:
        return "nr"
    return "lte"


def _get_selected_preamble_pre_test_condition(state: PipelineState) -> str:
    """Return preamble pre-test condition from selected context JSON."""
    context_json = state.get("selected_context_json") or {}
    if not isinstance(context_json, dict):
        return ""

    value = context_json.get("preamble_pre_test_condition")
    if not value:
        # Fallback in case upstream field naming uses a title-style key.
        value = context_json.get("Preamble pre test conditions")

    return str(value or "")


def _coerce_test_purpose_options(value: object) -> list[str]:
    """Normalize selected_context_json.test_purposes into a clean option list."""
    normalised: list[str] = []

    def _append_option(option: object) -> None:
        text = ""
        if isinstance(option, dict):
            for key in ("test_purpose", "purpose", "name", "title", "text", "description"):
                candidate = str(option.get(key, "")).strip()
                if candidate:
                    text = candidate
                    break
        else:
            text = str(option or "").strip()

        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
        if text and text not in normalised:
            normalised.append(text)

    if isinstance(value, list):
        for item in value:
            _append_option(item)
        return normalised

    text = str(value or "").strip()
    if not text or text in {"-", "none", "None", "[]"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                _append_option(item)
            return normalised

    line_items = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in re.split(r"\r?\n+", text)
        if re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
    ]
    if len(line_items) > 1:
        for item in line_items:
            _append_option(item)
        return normalised

    if ";" in text:
        for item in text.split(";"):
            _append_option(item)
        if normalised:
            return normalised

    marker_count = len(re.findall(r"(?:^|\s)\d+[.)]\s+", text))
    if marker_count >= 2:
        numbered_parts = [
            part.strip()
            for part in re.split(r"(?:^|\s)\d+[.)]\s+", text)
            if part.strip()
        ]
        for item in numbered_parts:
            _append_option(item)
        if len(normalised) > 1:
            return normalised

    _append_option(text)
    return normalised


def _get_test_purpose_value(context_json: dict) -> object:
    """Extract test-purpose field from schema variants used by LLM outputs."""
    if not isinstance(context_json, dict):
        return []
    for key in (
        "test_purposes",
        "test_purpose",
        "testPurpose",
        "test_purpose_list",
        "test purpose",
        "test purposes",
    ):
        if key in context_json and context_json.get(key) not in (None, ""):
            return context_json.get(key)
    return []


def _extract_transition_messages_from_table_context(*, rat: str, lookup: str, context: str) -> list[dict]:
    """Extract ordered transition messages from one LTE/NR table context."""
    lookup_text = str(lookup or "").strip()
    context_text = str(context or "").strip()
    if not context_text:
        return []

    try:
        llm = _get_llm()
        parser = _get_json_parser()
        chain = llm | parser
        if str(rat).strip().lower() == "nr":
            prompt = NR_RRC_TABLE_MESSAGE_PROMPT.format(
                rrc_lookup_query=lookup_text,
                table_context=context_text,
            )
        else:
            prompt = LTE_TRANSITION_TABLE_MESSAGE_PROMPT.format(
                transition_query=lookup_text,
                table_context=context_text,
            )

        parsed = chain.invoke(prompt)
        if not isinstance(parsed, dict):
            return []
        return _coerce_messages_payload(parsed.get("message_sequence", []))
    except Exception:
        return []


def _build_state_grouped_transition_messages(*, rat: str, loop_path: list[str]) -> tuple[list[dict], list[str], bool, str]:
    """Build per-state message groups directly from 36.508/38.508 retrieval payload.

    LTE rule: transition messages are assigned to destination states only.
    Example 1->2, 2->3 => groups for state 2 and state 3 (state 1 remains empty).
    """
    states = [str(s).strip() for s in (loop_path or []) if str(s).strip()]
    if not states:
        return [], [], False, "loop path is empty"

    try:
        retrieval_payload = retrieve_508_context.invoke({
            "rat": rat,
            "loop_path": " -> ".join(states),
        })
    except Exception as exc:
        return [], [], False, f"508 retrieval failed: {exc}"

    context_list: list[str] = []
    grouped: list[dict] = []

    def _append_group(state_name: str, messages: list[dict]) -> None:
        cleaned_state = str(state_name or "").strip()
        if not cleaned_state:
            return
        grouped.append({
            "state": cleaned_state,
            "messages": _coerce_messages_payload(messages),
        })

    rat_value = str(rat or "").strip().lower()
    if rat_value == "lte":
        sequence_chunks = retrieval_payload.get("sequence_chunks") or []
        for chunk in sequence_chunks:
            if not isinstance(chunk, dict):
                continue
            transition = str(chunk.get("transition", "")).strip()
            context_text = str(chunk.get("context", "")).strip()
            if context_text:
                context_list.append(context_text)

            # LTE transition rows belong to the destination state.
            if "->" in transition:
                _, to_state = transition.split("->", 1)
                destination_state = to_state.strip()
            else:
                destination_state = ""

            lookup = str(chunk.get("transition_query") or transition or destination_state).strip()
            messages = _extract_transition_messages_from_table_context(
                rat="lte",
                lookup=lookup,
                context=context_text,
            )
            _append_group(destination_state, messages)

        missing = retrieval_payload.get("missing_transitions") or []
        missing_text = ", ".join(str(item) for item in missing) if missing else ""
        complete = not bool(missing)
        error = f"missing LTE transition context for: {missing_text}" if missing_text else ""
    else:
        state_chunks = retrieval_payload.get("state_chunks") or []
        for chunk in state_chunks:
            if not isinstance(chunk, dict):
                continue
            state_id = str(chunk.get("state_id", "")).strip()
            context_text = str(chunk.get("context", "")).strip()
            if context_text:
                context_list.append(context_text)

            lookup = str(chunk.get("rrc_lookup_query") or state_id).strip()
            messages = _extract_transition_messages_from_table_context(
                rat="nr",
                lookup=lookup,
                context=context_text,
            )
            _append_group(state_id, messages)

        missing = retrieval_payload.get("missing_states") or []
        missing_text = ", ".join(str(item) for item in missing) if missing else ""
        complete = not bool(missing)
        error = f"missing NR state context for: {missing_text}" if missing_text else ""

    # Merge duplicate state buckets while preserving order.
    merged_by_state: dict[str, list[dict]] = {}
    state_order: list[str] = []
    for row in grouped:
        state_name = str(row.get("state", "")).strip()
        if not state_name:
            continue
        if state_name not in merged_by_state:
            merged_by_state[state_name] = []
            state_order.append(state_name)
        merged_by_state[state_name].extend(_coerce_messages_payload(row.get("messages", [])))

    merged_rows = [{"state": state_name, "messages": merged_by_state[state_name]} for state_name in state_order]
    return merged_rows, context_list, complete, error


def _extract_ue_transition_messages(
    *,
    state: PipelineState,
    ue_state: str,
) -> tuple[list[dict], str | None, bool, str | None, list[str], list[dict]]:
    rat = _resolve_rat_for_state_sequence(state)
    loop_map = _load_state_loop_map(rat)
    if not loop_map:
        return [], None, False, "state loop map not available", [], []

    state_key = _extract_state_key_from_ue_state(ue_state, rat)
    loop_path = loop_map.get(state_key) or loop_map.get(f"STATE {state_key}")
    if not loop_path:
        # No matching state found — return empty result list as success (no error)
        return [], None, True, None, [], []

    try:
        state_context_result: dict = retrieve_508_context.invoke({
            "rat": rat,
            "loop_path": loop_path,
        })
    except Exception as exc:
        return [], loop_path, False, f"failed to retrieve 508 context: {exc}", [], []

    transition_context = str(state_context_result.get("combined_context", "")).strip()
    if not transition_context:
        return [], loop_path, False, "retrieved 508 context is empty", [], []

    llm = _get_llm()
    parser = _get_json_parser()
    chain = llm | parser

    concatenated_messages: list[dict] = []
    transition_table_contexts: list[str] = []

    loop_states = [
        str(s).strip()
        for s in (state_context_result.get("loop_path") or [])
        if str(s).strip()
    ]
    state_message_map: dict[str, list[dict]] = {state_name: [] for state_name in loop_states}

    def _state_message_rows() -> list[dict]:
        rows = [{"state": name, "messages": msgs} for name, msgs in state_message_map.items()]
        # LTE transition extraction maps messages to destination state, so the
        # first loop state is expected to remain empty; hide it for cleaner output.
        if rat == "lte" and rows:
            rows = rows[1:]
        return rows

    if rat == "lte":
        sequence_chunks = state_context_result.get("sequence_chunks", [])
        missing_transitions = state_context_result.get("missing_transitions", [])
        expected_steps = max(0, len((state_context_result.get("loop_path") or [])) - 1)
        parsed_steps = 0
        for idx, item in enumerate(sequence_chunks):
            table_context = str(item.get("context", "")).strip()
            if not table_context:
                continue
            transition_table_contexts.append(table_context)
            transition_query = str(item.get("transition_query", "")).strip()
            if not transition_query:
                transition_query = str(item.get("transition", "")).strip()

            prompt = LTE_TRANSITION_TABLE_MESSAGE_PROMPT.format(
                transition_query=transition_query,
                table_context=table_context,
            )
            try:
                parsed = chain.invoke(prompt)
            except Exception:
                continue

            parsed_steps += 1

            extracted = parsed.get("message_sequence", []) if isinstance(parsed, dict) else []
            step_messages: list[dict] = []
            if isinstance(extracted, list):
                for msg in extracted:
                    if isinstance(msg, dict) and msg.get("name", "").strip():
                        step_messages.append(msg)
                    elif isinstance(msg, str) and msg.strip():
                        # legacy plain-string fallback
                        layer, clean_name = _split_layer_message(str(msg))
                        step_messages.append({"name": clean_name, "direction": "GNB_TO_UE", "layer": layer})

            if step_messages:
                concatenated_messages.extend(step_messages)
                # LTE extraction is transition-based; attach step messages to the destination
                # state. Extract destination state from the transition field (e.g., "2A->3A" -> "3A")
                # to handle cases where transitions are missing from the database.
                transition_str = str(item.get("transition", "")).strip()
                target_state = ""
                if "->" in transition_str:
                    parts = transition_str.split("->")
                    target_state = parts[-1].strip() if len(parts) > 1 else ""
                
                if target_state:
                    state_message_map.setdefault(target_state, []).extend(step_messages)

        if missing_transitions:
            return (
                concatenated_messages,
                loop_path,
                False,
                "missing LTE transition context for: " + ", ".join(str(t) for t in missing_transitions),
                transition_table_contexts,
                _state_message_rows(),
            )

        if expected_steps > 0 and parsed_steps < expected_steps:
            return (
                concatenated_messages,
                loop_path,
                False,
                f"parsed only {parsed_steps}/{expected_steps} LTE transition steps",
                transition_table_contexts,
                _state_message_rows(),
            )

        return (
            concatenated_messages,
            loop_path,
            True,
            None,
            transition_table_contexts,
            _state_message_rows(),
        )

    state_chunks = state_context_result.get("state_chunks", [])
    missing_states = state_context_result.get("missing_states", [])
    expected_steps = len(state_context_result.get("loop_path") or [])
    parsed_steps = 0
    for idx, item in enumerate(state_chunks):
        table_context = str(item.get("context", "")).strip()
        if not table_context:
            continue
        transition_table_contexts.append(table_context)
        rrc_lookup_query = str(item.get("rrc_lookup_query", "")).strip()
        if not rrc_lookup_query:
            rrc_lookup_query = str(item.get("rrc_state", "")).strip()

        prompt = NR_RRC_TABLE_MESSAGE_PROMPT.format(
            rrc_lookup_query=rrc_lookup_query,
            table_context=table_context,
        )
        try:
            parsed = chain.invoke(prompt)
        except Exception:
            continue

        parsed_steps += 1

        extracted = parsed.get("message_sequence", []) if isinstance(parsed, dict) else []
        step_messages: list[dict] = []
        if isinstance(extracted, list):
            for msg in extracted:
                if isinstance(msg, dict) and msg.get("name", "").strip():
                    step_messages.append(msg)
                elif isinstance(msg, str) and msg.strip():
                    layer, clean_name = _split_layer_message(str(msg))
                    step_messages.append({"name": clean_name, "direction": "GNB_TO_UE", "layer": layer})

        if step_messages:
            concatenated_messages.extend(step_messages)
             # For NR, use the state_id from the chunk itself rather than index
            # to handle cases where some states are missing from the database.
            state_name = str(item.get("state_id", "")).strip()
            if state_name:
                state_message_map[state_name].extend(step_messages)

    if missing_states:
        return (
            concatenated_messages,
            loop_path,
            False,
            "missing NR state context for: " + ", ".join(str(s) for s in missing_states),
            transition_table_contexts,
            _state_message_rows(),
        )

    if expected_steps > 0 and parsed_steps < expected_steps:
        return (
            concatenated_messages,
            loop_path,
            False,
            f"parsed only {parsed_steps}/{expected_steps} NR state steps",
            transition_table_contexts,
            _state_message_rows(),
        )

    return (
        concatenated_messages,
        loop_path,
        True,
        None,
        transition_table_contexts,
        _state_message_rows(),
    )


def _tp_matches_selected_test_purpose(tp_value: object, test_purpose_index: int | None) -> bool:
    """Return True when a message row should be included for selected test purpose.

    Rules:
    - TP with numeric ids => include only when selected id is present.
    - If selected id is missing/invalid => include all rows.
    """
    if not isinstance(test_purpose_index, int) or test_purpose_index <= 0:
        return True

    text = str(tp_value or "").strip()

    matches = re.findall(r"\d+", text)
    if not matches:
        return False

    target = str(test_purpose_index)
    return target in matches


def _filter_message_name_from_parameters(messages: list[dict]) -> list[dict]:
    """Keep only real parameters by removing items that equal the message name."""
    cleaned_messages: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        name = str(msg.get("name", "")).strip()
        name_key = re.sub(r"[^a-z0-9]", "", name.lower())

        params = _normalise_message_parameters(msg.get("message_parameters", []))
        filtered_params: list[str] = []
        for param in params:
            param_key = re.sub(r"[^a-z0-9]", "", str(param).lower())
            if not param_key:
                continue
            if name_key and param_key == name_key:
                continue
            if param not in filtered_params:
                filtered_params.append(param)

        updated = dict(msg)
        updated["message_parameters"] = filtered_params
        cleaned_messages.append(updated)

    return cleaned_messages

# Load config node
@traceable(name="load_config_node", run_type="parser",
           tags=["config"])
def load_config_node(state: PipelineState) -> dict:
    """Load the config JSON file specified by state["config_path"].
    """

    path = state["query_config_path"]
    with open(path) as f:
        config = json.load(f)
    spec_series_filter = _detect_spec_series_filter(config)
    return {
        "query_config": config,
        "initial_mandate_messages": [
            {"name": "MIB", "direction": "GNB_TO_UE", "cell_id": "", "layer": "SYSTEM"},
            {"name": "SIB1", "direction": "GNB_TO_UE", "cell_id": "", "layer": "SYSTEM"},
        ],
        "spec_series_filter": spec_series_filter,
    }


# Enhance query node
@traceable(name="enhance_query_node", run_type="retriever",
           tags=["retrieval", "rag"])
def enhance_query_node(state: PipelineState) -> dict:
    """Enhance the input question with retrieved RAG context.
    """
    try:
        # Top-k retrieval with the original question to get relevant context
        result: dict = retrieve_rag_context.invoke({
            "question": str(state["query_config"]),
            "top_k_search": QUERY_ENHANCE_TOP_SEARCH,
            "top_k_rerank": QUERY_ENHANCE_TOP_RERANK,
            "top_vector_wt": QUERY_ENHANCE_VECTOR_WEIGHT,
            "top_mmr_wt": QUERY_ENHANCE_MMR_WEIGHT,
            "top_bm25_wt": QUERY_ENHANCE_BM25_WEIGHT,
            "spec_series_filter": state.get("spec_series_filter"),
        })
        # Returns a list of top-k retrieved contexts (spec chunks) relevant to the original question.
        context_fetched = result.get("context", "")
        contexts = combine_context.invoke({
            "context": context_fetched
        })

        # Enhance the original question structure and format by appending the retrieved context
        enhanced_query: str = query_enhancement_rag.invoke({
            "query_config": str(state["query_config"]),
            "context": contexts
        })

        return {
            "question": enhanced_query
        }
    except Exception as exc:
        return {"error": f"enhance_query_node failed: {exc}"}


@traceable(name="direct_llm_answer_node", run_type="llm",
           tags=["extraction", "llm-only"])
def direct_llm_answer_node(state: PipelineState) -> dict:
    """Generate a direct LLM-only baseline answer from the enhanced question."""
    try:
        question = str(state.get("question", "")).strip()
        if not question:
            return {"error": "direct_llm_answer_node failed: question is empty"}

        llm = _get_llm()
        prompt = LLM_ONLY_PROMPT.format(question=question)
        raw = llm.invoke(prompt).content
        parsed = _parse_llm_only_response(raw)

        return {
            "llm_direct_message_sequence": parsed,
        }
    except Exception as exc:
        return {"error": f"direct_llm_answer_node failed: {exc}"}
    


@traceable(name="retrieve_context_options_node", run_type="retriever",
           tags=["retrieval", "rag"])
def retrieve_context_options_node(state: PipelineState) -> dict:
    """Retrieve top-k contexts and store them in state for downstream processing.

    Schema extraction is deferred to format_shortlisted_contexts_node so the
    more expensive LLM calls only run on the shortlisted subset, not every
    retrieved candidate.
    """
    try:
        result: dict = retrieve_rag_context.invoke({
            "question": str(state["question"]),
            "top_k_search": ANSWER_EXTRACT_TOP_SEARCH,
            "top_k_rerank": ANSWER_EXTRACT_TOP_RERANK,
            "top_vector_wt": ANSWER_EXTRACT_VECTOR_WEIGHT,
            "top_mmr_wt": ANSWER_EXTRACT_MMR_WEIGHT,
            "top_bm25_wt": ANSWER_EXTRACT_BM25_WEIGHT,
            "spec_series_filter": state.get("spec_series_filter"),
        })

        contexts = result.get("context", [])
        source_docs = result.get("source_doc", [])

        if not isinstance(contexts, list):
            return {"error": "retrieve_context_options_node failed: retrieved contexts must be a list"}

        return {
            "source_doc": source_docs,
            "raw_contexts": contexts,
            "raw_source_docs": source_docs,
        }
    except Exception as exc:
        return {"error": f"retrieve_context_options_node failed: {exc}"}


@traceable(name="shortlist_context_indices_node", run_type="llm",
           tags=["ranking", "rag"])
def shortlist_context_indices_node(state: PipelineState) -> dict:
    """DEPRECATED: merged with format_shortlisted_contexts_node. This now does nothing."""
    # This node is kept for compatibility but does nothing.
    # The work is done in shortlist_contexts_node instead.
    return {}


@traceable(name="shortlist_contexts_node", run_type="llm",
           tags=["ranking", "rag"])
def shortlist_contexts_node(state: PipelineState) -> dict:
    """Filter, rank, and extract schema fields from retrieved contexts using agentic shortlist agent.

    Merged functionality of:
    - shortlist_context_indices_node (LLM-based filtering)
    - format_shortlisted_contexts_node (schema extraction)

    The agent handles ranking and field extraction, returning ready-to-display options.
    """
    try:
        raw_contexts = state.get("raw_contexts", [])
        raw_source_docs = state.get("raw_source_docs", [])
        query_config = state.get("query_config", {})
        question = state.get("question", "")

        if not raw_contexts:
            return {"error": "shortlist_contexts_node failed: no raw_contexts available"}

        # Call the agentic shortlist agent
        result = run_shortlist_agent(
            raw_contexts=raw_contexts,
            raw_source_docs=raw_source_docs,
            query_config=query_config,
            question=question,
        )

        shortlisted_indices = result.get("shortlisted_raw_indices", [])
        shortlisted_enhanced_contexts = result.get("shortlisted_enhanced_contexts", [])
        formatted_options = result.get("shortlisted_comparison_json", [])

        if not shortlisted_indices:
            shortlisted_indices = list(range(min(5, len(raw_contexts))))
            shortlisted_enhanced_contexts = []
            formatted_options = []

        display_options = _build_shortlist_display_options(
            shortlisted_enhanced_contexts=shortlisted_enhanced_contexts,
            raw_contexts=raw_contexts,
            shortlisted_indices=shortlisted_indices,
        )

        print("\n" + "=" * 80)
        print("SHORTLISTED CONTEXT OPTIONS")
        print("=" * 80)
        if display_options:
            print(json.dumps(display_options, indent=2, ensure_ascii=False))
        print("\nSelect using displayed shortlist option_index (0..N-1).")
        print("=" * 80 + "\n")

        return {
            "shortlisted_raw_indices": shortlisted_indices,
            "shortlisted_enhanced_contexts": shortlisted_enhanced_contexts,
            "shortlisted_comparison_json": formatted_options,
            "shortlisted_display_options": display_options,
        }
    except Exception as exc:
        return {"error": f"shortlist_contexts_node failed: {exc}"}


def format_shortlisted_contexts_node(state: PipelineState) -> dict:
    """DEPRECATED: merged with shortlist_contexts_node via agentic agent. This now does nothing."""
    # This node is kept for compatibility but does nothing.
    # The work is done in shortlist_contexts_node instead.
    return {}


@traceable(name="select_context_node", run_type="parser",
           tags=["selection", "rag"])
def select_context_node(state: PipelineState) -> dict:
    """Pause for user option selection and persist the selected context in state.

    This node only resolves the user selection and stores stable inputs for
    downstream parallel extraction nodes.
    """
    contexts = state.get("raw_contexts", [])
    source_docs = state.get("raw_source_docs", [])
    shortlisted_indices = state.get("shortlisted_raw_indices", [])
    shortlisted_enhanced_contexts = state.get("shortlisted_enhanced_contexts", [])
    shortlisted_display_options = state.get("shortlisted_display_options", [])
    if not shortlisted_indices or not shortlisted_enhanced_contexts:
        return {"error": "select_context_node failed: no shortlisted options available"}

    display_options = shortlisted_display_options
    if not isinstance(display_options, list) or not display_options:
        display_options = _build_shortlist_display_options(
            shortlisted_enhanced_contexts=shortlisted_enhanced_contexts,
            raw_contexts=contexts,
            shortlisted_indices=shortlisted_indices,
        )

    # Pause here — surfaces enhanced_contexts to the caller.
    # Returns the value passed via Command(resume=<index>) on resume.
    selected_index = interrupt(display_options)

    try:
        if not isinstance(selected_index, int) or not (0 <= selected_index < len(shortlisted_indices)):
            return {
                "error": f"Invalid selected_context_index={selected_index}. Choose 0..{max(len(shortlisted_indices)-1, 0)}",
            }

        selected_raw_index = shortlisted_indices[selected_index]
        selected_context = contexts[selected_raw_index]
        final_context_user_choice = combine_context.invoke({"context": [selected_context]})
        selected_context_json = shortlisted_enhanced_contexts[selected_index].get("context_json", {})

        return {
            "selected_context_index": selected_index,
            "selected_raw_context_index": selected_raw_index,
            "rag_context": final_context_user_choice,
            "source_doc": source_docs,
            "selected_context_json": selected_context_json,
        }
    except Exception as exc:
        return {"error": f"select_context_node failed: {exc}"}


def _extract_section_id(source_doc: str) -> str | None:
    """Extract and normalize section hierarchy from source_doc."""
    if not source_doc or not isinstance(source_doc, str):
        return None
    parts = [part.strip() for part in source_doc.strip().split(">") if part.strip()]
    if len(parts) >= 2:
        return " > ".join(parts)
    return None


def _get_parent_section(section_id: str) -> str | None:
    """Get hierarchy parent by removing the final ' > child' segment."""
    if not section_id or ">" not in section_id:
        return None

    parts = [part.strip() for part in section_id.split(">") if part.strip()]
    if len(parts) < 3:
        return None

    return " > ".join(parts[:-1])


def _extract_section_numeric_prefix(section_id: str) -> tuple[int, ...]:
    """Extract sortable numeric prefix from the final hierarchy segment.

    Example final segment "8.1.4.1.2 Intra NR handover" -> (8, 1, 4, 1, 2)
    """
    if not section_id:
        return tuple()

    parts = [part.strip() for part in section_id.split(">") if part.strip()]
    if not parts:
        return tuple()

    last_segment = parts[-1]
    match = re.search(r"\b(\d+(?:\.\d+)*)\b", last_segment)
    if not match:
        return tuple()

    try:
        return tuple(int(p) for p in match.group(1).split("."))
    except ValueError:
        return tuple()


def _find_sibling_sections(
    section_id: str,
    source_docs: list[str],
    raw_contexts: list[str],
    spec_series_filter: str | None = None,
) -> list[dict]:
    """Find all sibling sections (including self) by scanning chunks.pkl directly.

    Uses find_sibling_chunks() from core.retrieval so all chunks under the same
    parent breadcrumb are returned, not just those that happened to be retrieved
    in the current RAG pass.

    Returns list of dicts:
        [{"section_id": "...", "summary": "...", "raw_index": <int|-1>,
          "corpus_context": "..."}, ...]
    raw_index is the index in raw_contexts/source_docs when the breadcrumb was
    already retrieved, otherwise -1.
    """
    if not section_id:
        return []

    # Build a fast lookup: normalised breadcrumb -> index in already-retrieved list.
    retrieved_index: dict[str, int] = {}
    for idx, doc in enumerate(source_docs):
        if doc:
            normalised = _extract_section_id(doc)
            if normalised:
                retrieved_index[normalised] = idx

    # Query the full corpus for siblings.
    sibling_chunks = find_sibling_chunks(
        breadcrumb=section_id,
        spec_series_filter=spec_series_filter,
    )

    if not sibling_chunks:
        # Corpus scan returned nothing — fall back to already-retrieved docs.
        parent = _get_parent_section(section_id)
        parent_parts = [p.strip() for p in parent.split(">") if p.strip()] if parent else []

        # Derive numeric parent prefix for secondary matching
        # e.g. selected "8.1.4.1.2 …" → parent_numeric (8,1,4,1)
        selected_parts = [p.strip() for p in section_id.split(">") if p.strip()]
        _sel_numeric = _extract_section_numeric_prefix(section_id)
        parent_numeric = _sel_numeric[:-1] if _sel_numeric else ()

        siblings: list[dict] = []
        seen_sections: set[str] = set()
        for idx, doc in enumerate(source_docs):
            if not doc or idx >= len(raw_contexts):
                continue
            doc_section = _extract_section_id(doc)
            if not doc_section or doc_section in seen_sections:
                continue
            doc_parts = [p.strip() for p in doc_section.split(">") if p.strip()]
            # Primary: structural descendant of parent path
            structural_match = (
                bool(parent_parts)
                and doc_parts[:len(parent_parts)] == parent_parts
                and len(doc_parts) > len(parent_parts)
            )
            # Secondary: numeric section starts with parent_numeric
            numeric_match = False
            if parent_numeric:
                doc_numeric = _extract_section_numeric_prefix(doc_section)
                numeric_match = (
                    len(doc_numeric) > len(parent_numeric)
                    and doc_numeric[:len(parent_numeric)] == parent_numeric
                )
            if structural_match or numeric_match:
                seen_sections.add(doc_section)
                siblings.append({
                    "section_id": doc_section,
                    "summary": _generate_summary(raw_contexts[idx]),
                    "raw_index": idx,
                    "corpus_context": raw_contexts[idx],
                })
    else:
        siblings = []
        for bc, context_text in sibling_chunks:
            raw_index = retrieved_index.get(bc, -1)
            # Use already-retrieved context when available; fall back to corpus text.
            ctx = raw_contexts[raw_index] if raw_index >= 0 else context_text
            siblings.append({
                "section_id": bc,
                "summary": _generate_summary(ctx),
                "raw_index": raw_index,
                "corpus_context": context_text,
            })


    return siblings


def _shortlist_sibling_indices(
    siblings: list[dict],
    query_config: dict,
    question: str,
) -> list[int]:
    """Use an LLM to keep only the most relevant sibling section indices.

    Returns a deduplicated list of sibling indices to show the user.
    Falls back to all indices when the LLM call fails.
    """
    if not siblings:
        return []

    indexed = [
        {"option_index": i, "section_id": s["section_id"], "context": s.get("corpus_context", "")}
        for i, s in enumerate(siblings)
    ]

    prompt = SIBLING_OPTION_SHORTLIST_PROMPT.format(
        query_config=json.dumps(query_config, indent=2),
        question=str(question),
        sibling_contexts_json=json.dumps(indexed, indent=2),
    )

    try:
        llm = _get_llm()
        parser = _get_json_parser()
        parsed = (llm | parser).invoke(prompt)
        selected = parsed.get("selected_option_indices", [])
        if not isinstance(selected, list):
            selected = []

        dedup: list[int] = []
        for idx in selected:
            if isinstance(idx, int) and 0 <= idx < len(siblings) and idx not in dedup:
                dedup.append(idx)

        return dedup if dedup else list(range(len(siblings)))
    except Exception:
        return list(range(len(siblings)))


def _generate_summary(context: str, max_chars: int = 300) -> str:
    """Extract a concise test objective from context using the LLM.

    Only the first 2000 chars are used to keep latency/cost bounded.
    """
    text = str(context or "").strip()
    if not text:
        return ""

    snippet = text[:2000]
    prompt = (
        "Extract the main test objective from the given 3GPP context.\n"
        "Return one concise plain-text sentence only (no bullets, no markdown, no preface).\n"
        f"Keep the response under {max_chars} characters.\n\n"
        "Context:\n"
        f"{snippet}"
    )

    try:
        llm = _get_llm()
        raw = llm.invoke(prompt).content
        objective = " ".join(str(raw or "").strip().split())
        if not objective:
            raise ValueError("empty objective")
    except Exception:
        objective = " ".join(line.strip() for line in text.splitlines() if line.strip())

    if len(objective) > max_chars:
        objective = objective[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return objective


def _derive_option_heading(context_json: object, source_doc: str, fallback: str) -> str:
    """Create a compact test heading for option-card displays."""
    if isinstance(context_json, dict):
        for key in ("test_name", "test_heading", "title", "name", "objective"):
            value = str(context_json.get(key, "")).strip()
            if value:
                return value

    source_text = str(source_doc or "").strip()
    if source_text:
        parts = [p.strip() for p in source_text.split(">") if p.strip()]
        if parts:
            return parts[-1]

    return str(fallback or "Option").strip() or "Option"


def _build_shortlist_display_options(
    shortlisted_enhanced_contexts: list[dict],
    raw_contexts: list[str],
    shortlisted_indices: list[int],
) -> list[dict]:
    """Return lightweight first-selection payloads with heading+objective only."""
    options: list[dict] = []
    for idx, item in enumerate(shortlisted_enhanced_contexts):
        if not isinstance(item, dict):
            continue

        raw_idx = shortlisted_indices[idx] if idx < len(shortlisted_indices) else -1
        raw_context = raw_contexts[raw_idx] if isinstance(raw_idx, int) and 0 <= raw_idx < len(raw_contexts) else ""

        context_json = item.get("context_json") if isinstance(item.get("context_json"), dict) else {}
        heading = _derive_option_heading(
            context_json=context_json,
            source_doc=str(item.get("source_doc", "")),
            fallback=f"Option {idx}",
        )
        objective = ""
        if isinstance(context_json, dict):
            objective = (
                str(context_json.get("objective", "")).strip()
                or str(context_json.get("test_objective", "")).strip()
                or str(context_json.get("aim", "")).strip()
                or str(context_json.get("summary", "")).strip()
            )
        if not objective:
            objective = _generate_summary(raw_context)

        options.append(
            {
                "option_index": idx,
                "test_heading": heading,
                "objective": objective,
                "summary": objective,
            }
        )
    return options


@lru_cache(maxsize=512)
def _extract_spec_reference_with_llm(text: str, purpose: str) -> tuple[str, str]:
    """Extract (spec_id, section_id) using LLM from reference context text."""
    payload = str(text or "").strip()
    if not payload:
        return ("-", "-")

    snippet = payload[:2000]
    prompt = (
        "You extract reference ids from 3GPP context.\n"
        "Return strict JSON with keys: spec_id, section_id.\n"
        "spec_id format example: 38.508 or 36.523\n"
        "section_id format example: 4.5.2 or 8.1\n"
        "If unavailable, use '-'.\n"
    )
    if purpose == "ue_transition":
        prompt += (
            "For UE transition references, prefer transition-table specification ids "
            "(e.g., 38.508 / 36.508) and clause-level section ids.\n"
        )
    else:
        prompt += (
            "For procedure references, use the final selected procedure section in context.\n"
        )
    prompt += f"\nContext:\n{snippet}"

    try:
        llm = _get_llm()
        parser = _get_json_parser()
        parsed = (llm | parser).invoke(prompt)
        spec_id = str(parsed.get("spec_id", "")).strip() or "-"
        section_id = str(parsed.get("section_id", "")).strip() or "-"
        return (spec_id, section_id)
    except Exception:
        return ("-", "-")


def _build_reference_lines(state: PipelineState) -> tuple[list[str], list[str], str]:
    """Create final UE transition and procedure references for result display."""
    ue_refs: list[str] = []
    procedure_refs: list[str] = []

    loop_path = state.get("ue_state_loop_path") or []
    if not isinstance(loop_path, list):
        loop_path = [str(loop_path).strip()] if str(loop_path).strip() else []

    transition_contexts = state.get("ue_transition_table_contexts") or []
    if not isinstance(transition_contexts, list):
        transition_contexts = [str(transition_contexts)] if str(transition_contexts).strip() else []

    fallback_doc = ""
    raw_source_docs = state.get("raw_source_docs") or []
    selected_raw_index = state.get("selected_raw_context_index")
    if isinstance(selected_raw_index, int) and 0 <= selected_raw_index < len(raw_source_docs):
        fallback_doc = str(raw_source_docs[selected_raw_index])
    if not fallback_doc:
        source_docs = state.get("source_doc") or []
        if isinstance(source_docs, list) and source_docs:
            fallback_doc = str(source_docs[0])

    spec_series = str(state.get("spec_series_filter") or "").strip()
    ue_transition_spec = "38.508" if spec_series == "38" else "36.508"

    for idx, loop_state in enumerate(loop_path):
        state_label = str(loop_state or "").strip() or "-"
        # Starting point has no transition reference by definition.
        if "start" in state_label.lower():
            continue
        matched_context = ""
        if idx < len(transition_contexts):
            matched_context = str(transition_contexts[idx] or "").strip()
        if not matched_context:
            for context in transition_contexts:
                text = str(context or "")
                if state_label and state_label.lower() in text.lower():
                    matched_context = text
                    break
        if not matched_context and transition_contexts:
            matched_context = str(transition_contexts[-1] or "")

        _spec_id, section_id = _extract_spec_reference_with_llm(
            matched_context or fallback_doc,
            "ue_transition",
        )
        ue_refs.append(f"{state_label} - {ue_transition_spec} $ {section_id}")

    proc_spec, proc_section = _extract_spec_reference_with_llm(fallback_doc, "procedure")
    # Keep procedure spec normalized to 3GPP dotted style without suffix noise.
    proc_match = re.search(r"\b(3[68](?:\.|)523)\b", str(proc_spec))
    if proc_match:
        token = proc_match.group(1).replace(".", "")
        proc_spec = f"{token[:2]}.{token[2:]}"
    else:
        proc_spec = "38.523" if spec_series == "38" else "36.523"
    procedure_refs.append(f"Procedure - {proc_spec} $ {proc_section}")

    lines: list[str] = []
    if ue_refs:
        lines.append("UE TRANSITION REFERENCES")
        lines.extend(ue_refs)
    if procedure_refs:
        if lines:
            lines.append("")
        lines.append("PROCEDURE REFERENCES")
        lines.extend(procedure_refs)

    return ue_refs, procedure_refs, "\n".join(lines).strip()


def _is_benign_empty_extraction(reason: object) -> bool:
    """Return True for expected 'no messages found' extraction outcomes."""
    text = str(reason or "").strip().lower()
    if not text:
        return False
    benign_markers = (
        "no transition messages extracted",
        "no messages extracted",
        "no messages found",
        "no relevant messages",
        "starting point",
        "start point",
    )
    return any(marker in text for marker in benign_markers)


@traceable(name="select_final_sibling_section_node", run_type="parser",
           tags=["selection", "sibling"])
def select_final_sibling_section_node(state: PipelineState) -> dict:
    """Find sibling sections of the selected context and ask user to make final selection.
    
    After the user selected one option from the shortlist, this node finds all
    sibling sections (including self) under the same parent section, displays
    them with brief summaries, and interrupts for the user to select the final
    section before the pipeline continues with extraction.
    """
    try:
        # GUARD: If sibling selection is already complete, skip re-processing
        if state.get("sibling_final_selection_index") is not None:
            return {}
        
        selected_raw_context_index = state.get("selected_raw_context_index")
        # Fallback: derive from selected_context_index + shortlisted_raw_indices
        # (handles cases where the value wasn't committed to the checkpoint yet)
        if not isinstance(selected_raw_context_index, int):
            selected_context_index = state.get("selected_context_index")
            shortlisted_indices = state.get("shortlisted_raw_indices", [])
            if isinstance(selected_context_index, int) and 0 <= selected_context_index < len(shortlisted_indices):
                selected_raw_context_index = shortlisted_indices[selected_context_index]

        raw_source_docs = state.get("raw_source_docs", [])
        raw_contexts = state.get("raw_contexts", [])
        spec_series_filter = state.get("spec_series_filter")

        if not isinstance(selected_raw_context_index, int):
            return {"error": "select_final_sibling_section_node: invalid selected_raw_context_index"}
        
        if selected_raw_context_index >= len(raw_source_docs):
            return {"error": "select_final_sibling_section_node: selected_raw_context_index out of range"}
        
        selected_source_doc = raw_source_docs[selected_raw_context_index]
        selected_section_id = _extract_section_id(selected_source_doc)
        
        if not selected_section_id:
            # No section ID found, treat as final selection
            return {
                "sibling_section_options": [],
                "sibling_sections_available": False,
            }
        
        # Find all siblings
        siblings = _find_sibling_sections(
            section_id=selected_section_id,
            source_docs=raw_source_docs,
            raw_contexts=raw_contexts,
            spec_series_filter=spec_series_filter,
        )
        
        if len(siblings) <= 1:
            # No siblings available, treat as final selection
            return {
                "sibling_section_options": siblings,
                "sibling_sections_available": False,
            }

        # LLM-shortlist siblings before presenting to the user.
        shortlisted_indices = _shortlist_sibling_indices(
            siblings=siblings,
            query_config=state.get("query_config") or {},
            question=str(state.get("question") or ""),
        )
        # Format shortlisted sibling options for display
        display_options = []
        for display_idx, orig_idx in enumerate(shortlisted_indices):
            section_id = str(siblings[orig_idx].get("section_id", "")).strip()
            heading = section_id.split(">")[-1].strip() if section_id else f"Sibling Option {display_idx}"
            objective = str(siblings[orig_idx].get("summary", "")).strip()
            display_options.append({
                "option_index": display_idx,
                "original_sibling_index": orig_idx,
                "test_heading": heading,
                "objective": objective,
                "summary": objective,
            })

        print("\n" + "="*80)
        print("SIBLING SECTIONS (Final Selection)")
        print("="*80)
        print(json.dumps(display_options, indent=2, ensure_ascii=False))
        print("\nSelect final section using option_index (0..N-1).")
        print("="*80 + "\n")

        # Interrupt for user selection
        final_index = interrupt(display_options)
        
        # Map user's display index back to the full siblings list index.
        resolved_final_index = shortlisted_indices[final_index] if isinstance(final_index, int) and 0 <= final_index < len(shortlisted_indices) else final_index

        return {
            "sibling_section_options": siblings,
            "sibling_sections_available": True,
            "sibling_final_selection_index": resolved_final_index,
            "sibling_display_options": display_options,
        }
    except Exception as exc:
        from langgraph.errors import GraphInterrupt as _GraphInterrupt
        if isinstance(exc, _GraphInterrupt):
            raise
        return {"error": f"select_final_sibling_section_node failed: {exc}"}


@traceable(name="finalize_section_selection_node", run_type="llm",
           tags=["selection", "sibling", "context-enhancement"])
def finalize_section_selection_node(state: PipelineState) -> dict:
    """Generate full context_enhancement_json for the finally selected sibling section.
    
    After the user selects from sibling options, this node:
    1. Loads the final selected context chunk
    2. Generates the full context_enhancement_json using LLM
    3. Updates state with the final selection
    """
    try:
        sibling_sections_available = state.get("sibling_sections_available", False)
        
        if not sibling_sections_available:
            # No sibling selection was made, return early with existing selection
            return {}
        
        sibling_final_selection_index = state.get("sibling_final_selection_index")
        sibling_section_options = state.get("sibling_section_options", [])
        raw_contexts = state.get("raw_contexts", [])
        raw_source_docs = state.get("raw_source_docs", [])
        
        if not isinstance(sibling_final_selection_index, int):
            return {"error": "finalize_section_selection_node: invalid sibling_final_selection_index"}
        
        if sibling_final_selection_index >= len(sibling_section_options):
            return {"error": "finalize_section_selection_node: sibling_final_selection_index out of range"}
        
        final_sibling = sibling_section_options[sibling_final_selection_index]
        final_raw_index = final_sibling.get("raw_index", -1)

        # Prefer already-retrieved context; fall back to corpus context stored on the sibling.
        if final_raw_index >= 0 and final_raw_index < len(raw_contexts):
            final_context = raw_contexts[final_raw_index]
            final_source_doc = raw_source_docs[final_raw_index] if final_raw_index < len(raw_source_docs) else ""
        else:
            corpus_context = final_sibling.get("corpus_context", "")
            if not corpus_context:
                return {"error": "finalize_section_selection_node: no context available for selected sibling"}
            final_context = corpus_context
            final_source_doc = final_sibling.get("section_id", "")
        
        # Generate full context_enhancement_json for the final selection
        context_json = generate_context_fields_json.invoke({"context": final_context})
        
        # Prepare final context
        final_context_user_choice = combine_context.invoke({"context": [final_context]})
        
        return {
            "selected_raw_context_index": final_raw_index,
            "rag_context": final_context_user_choice,
            "source_doc": raw_source_docs,
            "selected_context_json": context_json,
        }
    except Exception as exc:
        return {"error": f"finalize_section_selection_node failed: {exc}"}


@traceable(name="select_test_purpose_node", run_type="parser",
           tags=["selection", "test-purpose"])
def select_test_purpose_node(state: PipelineState) -> dict:
    """Pause for user test-purpose selection after the final context is resolved."""
    try:
        if state.get("test_purpose_index") is not None:
            return {}

        context_json = state.get("selected_context_json") or {}
        if not isinstance(context_json, dict):
            return {}

        test_purpose_options = _coerce_test_purpose_options(_get_test_purpose_value(context_json))
        if not test_purpose_options:
            return {}
        if len(test_purpose_options) == 1:
            return {"test_purpose_index": 1}

        display_options = [
            {"option_index": idx, "test_purpose": option}
            for idx, option in enumerate(test_purpose_options)
        ]

        print("\n" + "=" * 80)
        print("TEST PURPOSE OPTIONS")
        print("=" * 80)
        print(json.dumps(display_options, indent=2, ensure_ascii=False))
        print("\nSelect test purpose using option_index (0..N-1).")
        print("=" * 80 + "\n")

        selected_index = interrupt(display_options)
        if not isinstance(selected_index, int) or not (0 <= selected_index < len(test_purpose_options)):
            return {"error": "select_test_purpose_node failed: invalid test purpose selection"}

        return {"test_purpose_index": selected_index + 1}
    except Exception as exc:
        from langgraph.errors import GraphInterrupt as _GraphInterrupt
        if isinstance(exc, _GraphInterrupt):
            raise
        return {"error": f"select_test_purpose_node failed: {exc}"}


@traceable(name="extract_sib_messages_node", run_type="llm",
           tags=["extraction", "sib"])
def extract_sib_messages_node(state: PipelineState) -> dict:
    """Extract SIB messages from selected context fields using agentic SIB agent."""
    try:
        context_json = state.get("selected_context_json") or {}
        serving_cell, other_cells = get_cell_tuples(context_json, state)
        initial_mandate_messages = state.get("initial_mandate_messages") or []

        sib_messages: list[dict] = []
        # for cell in [serving_cell, *other_cells]:
        for cell in [serving_cell]:
            cell_id = str(cell.get("cell_id", "")).strip()
            if not cell_id:
                continue

            # Add initial mandatory broadcast messages for each participating cell
            # before appending that cell's SIB-state-specific messages.
            sib_messages.extend(
                assign_default_cell_id(
                    initial_mandate_messages,
                    cell_id,
                    override_existing=True,
                )
            )

            system_info_combination = normalise_sib_state(cell.get("sib_state", ""))
            if not system_info_combination:
                continue

            rat_label = infer_rat_label(cell.get("rat", ""), state.get("spec_series_filter"))
            rat = "lte" if rat_label == "LTE" else "nr"

            # Call agentic SIB agent
            cell_sib_messages = run_sib_agent(
                rat=rat,
                combination=system_info_combination,
                cell_id=cell_id,
            )
            cell_sib_messages = _coerce_messages_payload(cell_sib_messages)
            if not cell_sib_messages:
                continue

            sib_messages.extend(
                assign_default_cell_id(
                    cell_sib_messages,
                    cell_id,
                    override_existing=True,
                )
            )

        return {
            "sib_message_sequence": sib_messages,
            "sib_extraction_error": None,
        }
    except Exception as exc:
        # Some combinations may legitimately have no table rows/messages.
        if _is_benign_empty_extraction(exc):
            return {
                "sib_message_sequence": [],
                "sib_extraction_error": None,
            }
        return {"sib_extraction_error": f"extract_sib_messages_node failed: {exc}"}


@traceable(name="extract_ue_transition_messages_node", run_type="llm",
           tags=["extraction", "ue-state"])
def extract_ue_transition_messages_node(state: PipelineState) -> dict:
    """Extract UE state-transition messages using agentic UE transition agent."""
    try:
        context_json = state.get("selected_context_json") or {}
        serving_cell_id, other_cells = get_cell_metadata(context_json)

        def _to_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            text = str(value or "").strip().lower()
            return text in {"1", "true", "yes", "y", "ok", "complete", "completed", "success"}

        def _coerce_state_rows(value: object) -> list[dict]:
            if isinstance(value, list):
                rows: list[dict] = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    rows.append(
                        {
                            "state": str(item.get("state", "")).strip(),
                            "messages": _coerce_messages_payload(item.get("messages", [])),
                        }
                    )
                return rows
            if isinstance(value, dict):
                rows = []
                for state_name, messages in value.items():
                    rows.append(
                        {
                            "state": str(state_name).strip(),
                            "messages": _coerce_messages_payload(messages),
                        }
                    )
                return rows
            return []

        # Primary source requested by user: selected shortlist comparison JSON field.
        ue_state = _get_selected_preamble_pre_test_condition(state)

        # Fallbacks to keep compatibility if shortlist row is unavailable.
        if not ue_state.strip():
            ue_state = str(context_json.get("preamble_pre_test_condition", ""))
        if not ue_state.strip():
            ue_state = str(context_json.get("ue_state", ""))

        # For NR 38-series, extract state key from preamble.
        spec_series = str(state.get("spec_series_filter") or "").strip()
        if spec_series == "38":
            ue_state = _extract_state_key_from_ue_state(ue_state, "nr")

        # Determine RAT
        rat_label = infer_rat_label(context_json.get("rat", ""), spec_series)
        rat = "lte" if rat_label == "LTE" else "nr"

        # Call agentic UE transition agent
        result = run_ue_transition_agent(
            ue_state=ue_state,
            rat=rat,
            max_iterations=20,
        )

        if not isinstance(result, dict):
            result = {}

        transition_messages = _coerce_messages_payload(
            result.get("transition_message_sequence")
            or result.get("message_sequence")
            or result.get("ue_transition_message_sequence")
            or []
        )
        loop_path = result.get("loop_path") or result.get("loop_states") or []
        if not isinstance(loop_path, list):
            loop_path = [str(loop_path).strip()] if str(loop_path).strip() else []

        transition_table_contexts = result.get("table_contexts") or result.get("ue_transition_table_contexts") or []
        if not isinstance(transition_table_contexts, list):
            transition_table_contexts = [str(transition_table_contexts).strip()] if str(transition_table_contexts).strip() else []

        ue_transition_state_messages = _coerce_state_rows(
            result.get("state_messages")
            or result.get("state_message_map")
            or result.get("ue_transition_state_messages")
            or []
        )

        is_complete = _to_bool(result.get("is_complete", False))
        if not is_complete and transition_messages:
            # Agent produced concrete messages; treat as complete even if is_complete
            # flag is missing/serialized differently.
            is_complete = True

        incomplete_reason = str(result.get("error") or result.get("reason") or "").strip()

        # Ensure per-state grouping is available for downstream categorization.
        # For LTE, this maps transition i->j messages to destination state j,
        # keeping the first state intentionally empty.
        if loop_path and not ue_transition_state_messages:
            fallback_rows, fallback_contexts, fallback_complete, fallback_error = _build_state_grouped_transition_messages(
                rat=rat,
                loop_path=loop_path,
            )
            if fallback_rows:
                ue_transition_state_messages = fallback_rows
                transition_messages = [
                    msg
                    for row in fallback_rows
                    for msg in _coerce_messages_payload(row.get("messages", []))
                ]
                if fallback_contexts:
                    existing = [str(c).strip() for c in transition_table_contexts if str(c).strip()]
                    for ctx in fallback_contexts:
                        text = str(ctx).strip()
                        if text and text not in existing:
                            existing.append(text)
                    transition_table_contexts = existing
                if fallback_complete:
                    is_complete = True
                elif not incomplete_reason and fallback_error:
                    incomplete_reason = fallback_error

        if not is_complete:
            # Valid empty case: start-state or no transition rows are expected.
            if not transition_messages and not ue_transition_state_messages and (
                _is_benign_empty_extraction(incomplete_reason)
                or len(loop_path) <= 1
            ):
                return {
                    "ue_state_loop_path": loop_path,
                    "ue_transition_message_sequence": [],
                    "ue_transition_table_contexts": transition_table_contexts,
                    "ue_transition_state_messages": [],
                    "ue_transition_complete": True,
                    "ue_transition_error": None,
                }

            return {
                "ue_state_loop_path": loop_path,
                "ue_transition_message_sequence": assign_default_cell_id(
                    filter_sib_system_messages(transition_messages),
                    serving_cell_id,
                    override_existing=True,
                ),
                "ue_transition_table_contexts": transition_table_contexts,
                "ue_transition_state_messages": [
                    {
                        "state": item.get("state", ""),
                        "messages": assign_default_cell_id(
                            filter_sib_system_messages(item.get("messages", [])),
                            serving_cell_id,
                            override_existing=True,
                        ),
                    }
                    for item in ue_transition_state_messages
                ],
                "ue_transition_complete": False,
                "ue_transition_error": (
                    "extract_ue_transition_messages_node incomplete: "
                    f"{incomplete_reason or 'unknown reason'}"
                ),
            }

        return {
            "ue_state_loop_path": loop_path,
            "ue_transition_message_sequence": assign_default_cell_id(
                filter_sib_system_messages(transition_messages),
                serving_cell_id,
                override_existing=True,
            ),
            "ue_transition_table_contexts": transition_table_contexts,
            "ue_transition_state_messages": [
                {
                    "state": item.get("state", ""),
                    "messages": assign_default_cell_id(
                        filter_sib_system_messages(item.get("messages", [])),
                        serving_cell_id,
                        override_existing=True,
                    ),
                }
                for item in ue_transition_state_messages
            ],
            "ue_transition_complete": True,
            "ue_transition_error": None,
        }
    except Exception as exc:
        return {"ue_transition_error": f"extract_ue_transition_messages_node failed: {exc}"}


@traceable(name="extract_procedure_messages_node", run_type="llm",
           tags=["extraction", "procedure"])
def extract_procedure_messages_node(state: PipelineState) -> dict:
    """Extract scenario/procedure message sequence using agentic procedure agent with self-validation."""
    try:
        context_json = state.get("selected_context_json") or {}
        serving_cell_id, other_cells = get_cell_metadata(context_json)
        test_purpose_index = state.get("test_purpose_index")

        # Call agentic procedure agent (includes self-validation and retry logic)
        messages: list[dict] = run_procedure_agent(
            question=str(state.get("question", "")),
            context=str(state.get("rag_context", "")),
            serving_cell_id=serving_cell_id,
            other_cells=other_cells,
            test_purpose_index=int(test_purpose_index) if isinstance(test_purpose_index, int) else 0,
        )
        messages = _coerce_messages_payload(messages)

        filtered: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            cleaned = dict(message)
            cleaned.pop("tp", None)
            filtered.append(cleaned)

        # Apply parameter cleanup as a separate post-processing pass.
        filtered = _filter_message_name_from_parameters(filtered)
        filtered = filter_sib_system_messages(filtered)
        # Final cleanup: keep only rows with concrete message names.
        filtered = [
            msg
            for msg in filtered
            if isinstance(msg, dict) and str(msg.get("name", "")).strip()
        ]
        return {
            "procedure_message_sequence": assign_default_cell_id(filtered, serving_cell_id),
            "procedure_extraction_error": None,
        }
    except Exception as exc:
        # Empty procedure tables are valid for some selected contexts.
        if _is_benign_empty_extraction(exc):
            context_json = state.get("selected_context_json") or {}
            serving_cell_id, _ = get_cell_metadata(context_json)
            return {
                "procedure_message_sequence": assign_default_cell_id([], serving_cell_id),
                "procedure_extraction_error": None,
            }
        return {"procedure_extraction_error": f"extract_procedure_messages_node failed: {exc}"}


@traceable(name="concatenate_messages_node", run_type="parser",
           tags=["extraction", "final"])
def concatenate_messages_node(state: PipelineState) -> dict:
    """Concatenate all extracted message streams into one final ordered list."""
    try:
        if state.get("error"):
            # Preserve upstream error and avoid publishing a misleading final sequence.
            return {}

        branch_errors = [
            str(state.get("sib_extraction_error") or "").strip(),
            str(state.get("ue_transition_error") or "").strip(),
            str(state.get("procedure_extraction_error") or "").strip(),
        ]
        branch_errors = [msg for msg in branch_errors if msg]
        if branch_errors:
            return {"error": " | ".join(branch_errors)}

        context_json = state.get("selected_context_json") or {}
        serving_cell_id, _ = _get_cell_metadata(context_json)

        sib_messages = _assign_default_cell_id(
            state.get("sib_message_sequence") or [],
            serving_cell_id,
            override_existing=False,
        )
        transition_messages = _assign_default_cell_id(
            state.get("ue_transition_message_sequence") or [],
            serving_cell_id,
            override_existing=True,
        )
        procedure_messages = _assign_default_cell_id(state.get("procedure_message_sequence") or [], serving_cell_id)

        final_message_sequence = (
            sib_messages
            + transition_messages
            + procedure_messages
        )

        return {"message_sequence": final_message_sequence}
    except Exception as exc:
        return {"error": f"concatenate_messages_node failed: {exc}"}


@traceable(name="generate_sequence_diagram_node", run_type="tool",
           tags=["visualization", "final"])
def generate_sequence_diagram_node(state: PipelineState) -> dict:
    """Generate a grouped JPG sequence diagram and CSV for final messages.

    Categories:
    - SYSTEM INFORMATION COMBINATION: per-cell initial mandate + SIB messages
    - <state>: messages grouped by `ue_transition_state_messages[*].state`
    - PROCEDURE: procedure/scenario extraction messages
    """
    try:
        selected_context_json = state.get("selected_context_json") or {}
        serving_cell_id, participating_cell_ids = _get_cell_metadata(selected_context_json)

        system_category = "SYSTEM INFORMATION COMBINATION"

        sib_messages = _assign_default_cell_id(
            state.get("sib_message_sequence") or [],
            serving_cell_id,
            override_existing=False,
        )
        transition_messages = _assign_default_cell_id(
            state.get("ue_transition_message_sequence") or [],
            serving_cell_id,
            override_existing=True,
        )
        procedure_messages = _assign_default_cell_id(
            state.get("procedure_message_sequence") or [],
            serving_cell_id,
        )

        state_group_rows: list[dict] = []
        for row in state.get("ue_transition_state_messages") or []:
            state_label = str(row.get("state", "")).strip()
            category = f"UE STATE {state_label}" if state_label else "UE STATE"
            messages = _assign_default_cell_id(row.get("messages", []), serving_cell_id, override_existing=True)
            for msg in messages:
                state_group_rows.append({"category": category, "message": msg})

        categorized_rows: list[dict] = []
        categorized_rows.extend({"category": system_category, "message": msg} for msg in sib_messages)
        if state_group_rows:
            categorized_rows.extend(state_group_rows)
        else:
            categorized_rows.extend({"category": "STATE", "message": msg} for msg in transition_messages)
        categorized_rows.extend({"category": "PROCEDURE", "message": msg} for msg in procedure_messages)

        # Fallback to final sequence if categorized rows are unavailable.
        if not categorized_rows:
            fallback = _normalise_message_sequence(state.get("message_sequence") or [])
            categorized_rows = [{"category": "UNCLASSIFIED", "message": msg} for msg in fallback]

        if not categorized_rows:
            return {"sequence_diagram_file": "", "sequence_diagram_text": "", "sequence_csv_file": ""}

        normalised = [row["message"] for row in categorized_rows]

        # Build lane order: UE first, then serving cell, then participating cells,
        # then any extra cell IDs found in message payloads.
        lane_cells: list[str] = []
        for cell in [serving_cell_id, *participating_cell_ids]:
            cell_text = str(cell).strip()
            if cell_text and cell_text not in lane_cells:
                lane_cells.append(cell_text)

        normalised = _assign_default_cell_id(normalised, serving_cell_id)
        for msg in normalised:
            cell_text = str(msg.get("cell_id", "")).strip()
            if cell_text and cell_text not in lane_cells:
                lane_cells.append(cell_text)

        if not lane_cells:
            lane_cells = [str(serving_cell_id or "SCell1").strip() or "SCell1"]

        lane_labels = ["UE", *lane_cells]

        # Build category segments for grouped background rendering.
        segments: list[dict] = []
        seg_start = 1
        current_cat = str(categorized_rows[0].get("category", "UNCLASSIFIED"))
        for idx, row in enumerate(categorized_rows, start=1):
            cat = str(row.get("category", "UNCLASSIFIED"))
            if cat != current_cat:
                segments.append({"category": current_cat, "start": seg_start, "end": idx - 1})
                current_cat = cat
                seg_start = idx
        segments.append({"category": current_cat, "start": seg_start, "end": len(categorized_rows)})

        palette = ["#B9FFC7", "#B9D1FF", "#FFEBBE", "#FDC0C0", "#D7BCFF", "#BCF4FF"]
        category_colors: dict[str, str] = {}
        for i, segment in enumerate(segments):
            category_colors.setdefault(str(segment["category"]), palette[i % len(palette)])

        # Build compact text summary for logs and state inspection.
        diagram_lines: list[str] = []
        for idx, row in enumerate(categorized_rows, 1):
            msg = row["message"]
            category = str(row.get("category", "UNCLASSIFIED"))
            name = str(msg.get("name", "Unknown")).strip()
            layer = str(msg.get("layer", "")).strip().upper() or _infer_message_layer(name)
            direction = str(msg.get("direction", "GNB_TO_UE")).strip().upper()
            cell_id = str(msg.get("cell_id", "")).strip()
            params = _normalise_message_parameters(msg.get("message_parameters", []))
            params_suffix = f" | params: {', '.join(params)}" if params else ""
            display = f"{layer} | {name}"

            if direction == "UE_TO_GNB":
                detail = f"{idx:02d}. [{category}] UE -> {cell_id or serving_cell_id}: {display}{params_suffix}"
            elif direction == "GNB_TO_UE":
                detail = f"{idx:02d}. [{category}] {cell_id or serving_cell_id} -> UE: {display}{params_suffix}"
            else:
                detail = f"{idx:02d}. [{category}] {cell_id or serving_cell_id} <-> UE: {display}{params_suffix}"
            diagram_lines.append(detail)

        

        total = len(normalised)
        fig_height = max(6.0, min(0.9 * total + 3.5, 42.0))
        fig, ax = plt.subplots(figsize=(17, fig_height), dpi=220)

        # Background gradient for improved readability in reports.
        import numpy as np
        grad = np.linspace(0, 1, 400)
        gradient = np.vstack((grad, grad))
        ax.imshow(
            gradient,
            extent=[0, 1, 0, 1],
            transform=ax.transAxes,
            cmap="Blues",
            alpha=0.14,
            aspect="auto",
            zorder=0,
        )

        y_top = total + 1
        x_positions = np.linspace(0.08, 0.92, num=len(lane_labels))
        lane_x = {label: float(x) for label, x in zip(lane_labels, x_positions)}

        # Grouped colored background bands by category.
        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            category = str(segment["category"])
            y_upper = y_top - start + 0.5
            y_lower = y_top - end - 0.5
            band_color = category_colors.get(category, "#F3F4F6")
            ax.axhspan(y_lower, y_upper, facecolor=band_color, alpha=0.55, zorder=1)
            ax.text(
                0.03,
                (y_lower + y_upper) / 2,
                category,
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#334155",
                zorder=2,
            )

        # Actor headers and lifelines.
        for i, label in enumerate(lane_labels):
            is_ue = label == "UE"
            color = "#0B3D91" if is_ue else "#7A1E1E"
            line_color = "#2E4E8A" if is_ue else "#8C3A3A"
            x = lane_x[label]

            ax.text(
                x,
                y_top + 0.6,
                label,
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=color,
            )
            ax.plot([x, x], [0.3, y_top + 0.35], linestyle="--", linewidth=1.8, color=line_color, alpha=0.6)
            ax.add_patch(Circle((x, y_top + 0.1), 0.012, transform=ax.transData, color=line_color, zorder=5))

        for idx, row in enumerate(categorized_rows, 1):
            msg = row["message"]
            y = y_top - idx
            name = str(msg.get("name", "Unknown")).strip() or "Unknown"
            layer = str(msg.get("layer", "")).strip().upper() or _infer_message_layer(name)
            direction = str(msg.get("direction", "GNB_TO_UE")).strip().upper() or "GNB_TO_UE"
            cell_id = str(msg.get("cell_id", "")).strip() or serving_cell_id
            params = _normalise_message_parameters(msg.get("message_parameters", []))
            params_text = ", ".join(params)
            if len(params_text) > 80:
                params_text = params_text[:77].rsplit(",", 1)[0].strip() + "..."
            label_text = f"{layer} | {name}" if not params_text else f"{layer} | {name}\nparams: {params_text}"

            if cell_id not in lane_x:
                cell_id = lane_cells[0]

            if direction == "UE_TO_GNB":
                start_x, end_x = lane_x["UE"], lane_x[cell_id]
                color = "#1F7A8C"
            elif direction == "GNB_TO_UE":
                start_x, end_x = lane_x[cell_id], lane_x["UE"]
                color = "#C44536"
            else:
                # Unknown direction defaults to cell -> UE to keep output deterministic.
                start_x, end_x = lane_x[cell_id], lane_x["UE"]
                color = "#6A4C93"

            ax.annotate(
                "",
                xy=(end_x, y),
                xytext=(start_x, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=2.2,
                    color=color,
                    shrinkA=12,
                    shrinkB=12,
                    mutation_scale=18,
                ),
                zorder=3,
            )
            ax.text(
                (start_x + end_x) / 2,
                y + 0.13,
                label_text,
                ha="center",
                va="bottom",
                fontsize=10,
                color="#14213D",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FBFF", edgecolor="#BFD7EA", alpha=0.95),
                zorder=4,
            )

        ax.set_xlim(0.02, 0.98)
        ax.set_ylim(0, y_top + 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "Message Sequence Flow Diagram",
            fontsize=20,
            fontweight="bold",
            color="#0F2A43",
            pad=16,
        )
        ax.text(
            0.5,
            y_top + 0.25,
            f"Total Messages: {total} | Entities: {len(lane_labels)} | Categories: {len(segments)}",
            ha="center",
            va="center",
            fontsize=12,
            color="#334E68",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        ue_refs, procedure_refs, references_text = _build_reference_lines(state)
        if references_text:
            diagram_lines.append("")
            diagram_lines.append(references_text)

        diagram_text = "\n".join(diagram_lines)

        # Ensure visualizations directory exists
        viz_dir = Path(VISUALIZATIONS_DIR)
        eval_dir = Path(EVALUATION_DIR)
        viz_dir.mkdir(parents=True, exist_ok=True)

        config_stem = Path(str(QUERY_CONFIG_PATH)).stem.strip()
        # Keep filenames filesystem-safe while preserving readable config identity.
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", config_stem) or "query_config"

        suffix = 1
        while True:
            diagram_file = viz_dir / f"{safe_stem}_sequence_diagram_{suffix}.jpg"
            csv_file = eval_dir / f"{safe_stem}_sequence_messages_{suffix}.csv"
            if not diagram_file.exists() and not csv_file.exists():
                break
            suffix += 1

        # Ensure evaluation directory exists
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Write CSV output (serial no. | cell | direction | layer | message name | message parameters | category)
        with csv_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            test_name = str(selected_context_json.get("test_name", "")).strip() or "-"
            objective = str(selected_context_json.get("objective", "")).strip() or "-"
            test_purpose_index = state.get("test_purpose_index")
            test_purpose = str(test_purpose_index) if isinstance(test_purpose_index, int) and test_purpose_index > 0 else "-"
            writer.writerow(["TEST NAME", test_name])
            writer.writerow(["OBJECTIVE", objective])
            writer.writerow(["TEST PURPOSE", test_purpose])
            writer.writerow([])
            writer.writerow(["STEP", "CELL", "DIRECTION", "LAYER", "MESSAGE NAME", "MESSAGE PARAMETERS", "CATEGORY"])
            for idx, row in enumerate(categorized_rows, start=1):
                msg = row["message"]
                writer.writerow([
                    idx,
                    str(msg.get("cell_id", "") or serving_cell_id),
                    str(msg.get("direction", "") or "UNKNOWN"),
                    str(msg.get("layer", "") or _infer_message_layer(str(msg.get("name", "")))),
                    str(msg.get("name", "")),
                    "; ".join(_normalise_message_parameters(msg.get("message_parameters", []))),
                    str(row.get("category", "UNCLASSIFIED")),
                ])

        # Write diagram image to file
        fig.savefig(diagram_file, format="jpg", dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)

        print(f"\n[Sequence Diagram] Generated → {diagram_file}")
        print(f"[Sequence Diagram] CSV Export → {csv_file}")

        return {
            "sequence_diagram_file": str(diagram_file),
            "sequence_diagram_text": diagram_text,
            "sequence_csv_file": str(csv_file),
            "ue_transition_references": ue_refs,
            "procedure_references": procedure_refs,
            "final_references_text": references_text,
        }
    except Exception as exc:
        return {"error": f"generate_sequence_diagram_node failed: {exc}"}



