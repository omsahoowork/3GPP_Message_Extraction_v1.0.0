"""Shared utility functions for message normalization, cell ID resolution, and parameter parsing.

These helpers are extracted from nodes.py to be reused across nodes, agents, and tools.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph_pipeline.state import PipelineState


def infer_message_layer(name: str) -> str:
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


def split_layer_message(raw_name: str, explicit_layer: str = "") -> tuple[str, str]:
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
                return candidate_layer or infer_message_layer(message_name), message_name
    return (layer_text or infer_message_layer(name_text), name_text)


def normalise_cell_id(cell_id: object) -> str:
    """Canonicalize cell identifiers so spacing differences do not create new lanes."""
    text = str(cell_id or "").strip()
    return re.sub(r"\s+", "", text)


def normalise_message_parameters(value: object) -> list[str]:
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
            return normalise_message_parameters(parsed)

    parts = [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
    dedup: list[str] = []
    for part in parts:
        if part not in dedup:
            dedup.append(part)
    return dedup


def normalise_message_sequence(items: object, default_direction: str = "GNB_TO_UE") -> list[dict]:
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
            cell_id = normalise_cell_id(item.get("cell_id", ""))
            layer, clean_name = split_layer_message(name, str(item.get("layer", "")))
            message_parameters = normalise_message_parameters(item.get("message_parameters", []))
            normalised.append({
                "name": clean_name,
                "direction": direction,
                "cell_id": cell_id,
                "layer": layer,
                "message_parameters": message_parameters,
            })
        elif isinstance(item, str) and item.strip():
            layer, clean_name = split_layer_message(item.strip())
            normalised.append({
                "name": clean_name,
                "direction": default_direction,
                "cell_id": "",
                "layer": layer,
                "message_parameters": [],
            })
    return normalised


def infer_rat_label(value: object, fallback: str = "") -> str:
    """Infer RAT label (NR/LTE) from text or fallback."""
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


def normalise_sib_state(value: object) -> str:
    """Normalize SIB state/combination text."""
    text = str(value or "").strip()
    if not text or text in {"-", "none", "None", "null", "[]"}:
        return ""
    return re.sub(r"\s+", " ", text)


def infer_rat_from_sib_state(sib_state: str, fallback_rat: str = "") -> str:
    """Infer RAT from SIB state text."""
    text = str(sib_state or "").strip().upper()
    if not text:
        return infer_rat_label("", fallback_rat)
    if "NR-" in text or re.search(r"\bNR\b", text):
        return "NR"
    if "SYSTEM INFORMATION COMBINATION" in text or re.search(r"\bE-UTRA\b|\bLTE\b", text):
        return "LTE"
    return infer_rat_label(text, fallback_rat)


def split_tuple_like_text(value: str) -> list[str]:
    """Parse tuple-like strings: (a,b) or [a,b] or a,b."""
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


def coerce_cell_tuple(value: object, fallback_sib: str = "", fallback_rat: str = "") -> dict:
    """Normalize cell metadata into {cell_id, sib_state, rat}."""
    empty = {"cell_id": "", "sib_state": "", "rat": infer_rat_label("", fallback_rat)}

    if isinstance(value, dict):
        cell_id = normalise_cell_id(
            value.get("cell_id")
            or value.get("cell")
            or value.get("id")
            or value.get("name")
            or ""
        )
        sib_state = normalise_sib_state(
            value.get("sib_state")
            or value.get("system_information_combination")
            or value.get("system_information_combinations")
            or fallback_sib
        )
        rat = infer_rat_label(value.get("rat") or cell_id, fallback_rat)
        return {"cell_id": cell_id, "sib_state": sib_state, "rat": rat}

    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        parts = split_tuple_like_text(str(value or ""))

    if not parts:
        return empty

    cell_id = normalise_cell_id(parts[0]) if len(parts) >= 1 else ""
    sib_state = normalise_sib_state(parts[1]) if len(parts) >= 2 else normalise_sib_state(fallback_sib)
    rat_source = parts[2] if len(parts) >= 3 else cell_id
    rat = infer_rat_label(rat_source or cell_id, fallback_rat)
    return {"cell_id": cell_id, "sib_state": sib_state, "rat": rat}


def coerce_cell_tuple_list(value: object, fallback_sib: str = "", fallback_rat: str = "") -> list[dict]:
    """Coerce mixed scalar/list cell-id values into a clean list of tuples."""
    if isinstance(value, list):
        if value and not any(isinstance(item, (list, dict)) for item in value):
            single = coerce_cell_tuple(value, fallback_sib=fallback_sib, fallback_rat=fallback_rat)
            return [single] if single.get("cell_id") else []

        tuples: list[dict] = []
        for item in value:
            cell_tuple = coerce_cell_tuple(item, fallback_sib=fallback_sib, fallback_rat=fallback_rat)
            if cell_tuple.get("cell_id"):
                tuples.append(cell_tuple)
        return tuples

    single = coerce_cell_tuple(value, fallback_sib=fallback_sib, fallback_rat=fallback_rat)
    return [single] if single.get("cell_id") else []


def get_default_rat_hint(context_json: dict, state: PipelineState | None = None) -> str:
    """Infer RAT from context JSON or state."""
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


def get_cell_tuples(context_json: dict, state: PipelineState | None = None) -> tuple[dict, list[dict]]:
    """Resolve serving and participating cells as {cell_id, sib_state, rat}."""
    if not isinstance(context_json, dict):
        return {"cell_id": "SCell1", "sib_state": "", "rat": ""}, []

    fallback_rat = get_default_rat_hint(context_json, state)

    serving = coerce_cell_tuple(
        context_json.get("serving_cell_id_or_number", ""),
        fallback_sib="",
        fallback_rat=fallback_rat,
    )
    if serving.get("cell_id", "").lower() in {"", "none", "null", "-"}:
        serving = {"cell_id": "SCell1", "sib_state": serving.get("sib_state", ""), "rat": serving.get("rat", fallback_rat)}

    others = coerce_cell_tuple_list(
        context_json.get("other_participating_cell_id_or_number_list", []),
        fallback_sib="",
        fallback_rat=fallback_rat,
    )

    serving_id = serving.get("cell_id", "")
    if serving_id:
        others = [cell for cell in others if cell.get("cell_id") != serving_id]

    global_sib_state = normalise_sib_state(context_json.get("system_information_combinations", ""))
    if global_sib_state:
        global_sib_rat = infer_rat_from_sib_state(global_sib_state, fallback_rat)
        all_cells = [serving, *others]

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
                if not normalise_sib_state(cell.get("sib_state", "")) and str(cell.get("rat", "")).upper() == global_sib_rat:
                    cell["sib_state"] = global_sib_state

    return serving, others


def get_cell_metadata(context_json: dict) -> tuple[str, list[str]]:
    """Resolve serving and participating cell IDs from selected context JSON."""
    serving, others = get_cell_tuples(context_json)
    return serving.get("cell_id", "SCell1") or "SCell1", [cell.get("cell_id", "") for cell in others if cell.get("cell_id")]


def assign_default_cell_id(
    messages: object,
    serving_cell_id: str,
    *,
    override_existing: bool = False,
) -> list[dict]:
    """Ensure each message dict contains cell_id with serving-cell fallback."""
    def _resolve_cell_id(value: object, fallback: str) -> str:
        text = normalise_cell_id(value)
        if not text:
            return fallback

        lowered = re.sub(r"[^a-z0-9]", "", str(text).lower())
        if lowered in {"", "none", "null", "na", "unknown", "servingcellid", "servingcell", "defaultcellid"}:
            return fallback
        if "servingcell" in lowered:
            return fallback

        return text

    default_cell = normalise_cell_id(serving_cell_id)
    normalised = normalise_message_sequence(messages)
    updated: list[dict] = []
    for msg in normalised:
        current = _resolve_cell_id(msg.get("cell_id", ""), default_cell)
        layer = str(msg.get("layer", "")).strip().upper() or infer_message_layer(str(msg.get("name", "")))
        updated.append({
            "name": msg.get("name", ""),
            "direction": msg.get("direction", "GNB_TO_UE"),
            "cell_id": default_cell if override_existing else (current or default_cell),
            "layer": layer,
            "message_parameters": normalise_message_parameters(msg.get("message_parameters", [])),
        })
    return updated


_SIB_SYSTEM_INFO_PATTERN = re.compile(
    r"^SIB\d*$|SYSTEM\s+INFORMATION",
    flags=re.IGNORECASE,
)


def filter_sib_system_messages(messages: list[dict]) -> list[dict]:
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
