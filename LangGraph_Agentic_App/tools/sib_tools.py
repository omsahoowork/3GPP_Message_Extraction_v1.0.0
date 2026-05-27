"""Tools for SIB message extraction and lookup."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from core.utils import normalise_sib_state


@tool
def sib_lookup_table_text(
    rat: Annotated[str, "RAT type: 'nr' or 'lte'"],
) -> str:
    """Load the full SIB lookup table for the given RAT.

    Returns the complete markdown text for direct LLM reasoning.

    Args:
        rat: Either 'nr' or 'lte'

    Returns:
        Full SIB lookup table markdown text, or empty string if not found.
    """
    rat_value = str(rat).strip().lower()
    sib_filename = "nr_sib.md" if rat_value == "nr" else "lte_sib.md"
    sib_file = Path(__file__).parent.parent / "data_sib" / sib_filename
    if sib_file.exists():
        return sib_file.read_text(encoding="utf-8")
    return ""


def normalise_system_info_combination(rat: str, value: str) -> str:
    """Normalize system information combination text for lookup."""
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


def extract_sib_messages_from_sequence_text(sequence_text: str) -> list[str]:
    """Extract SIB message names from a sequence text string."""
    return re.findall(r"\bSIB\d+\b", str(sequence_text or ""), flags=re.IGNORECASE)


@tool
def lookup_sib_combination(
    rat: Annotated[str, "RAT type: 'nr' or 'lte'"],
    combination: Annotated[str, "System information combination e.g. 'NR-2' or '31'"],
) -> list[dict]:
    """Look up SIB message sequence for a given system information combination.

    Returns a direct table lookup from data_sib/nr_sib.md or data_sib/lte_sib.md.
    If the combination is not found, returns an empty list.

    Args:
        rat: Either 'nr' or 'lte'
        combination: The system information combination identifier

    Returns:
        List of dicts with {name, direction, layer, cell_id} for each SIB, or []
    """
    if not combination or combination == "-":
        return []

    rat_value = str(rat).strip().lower()
    sib_filename = "nr_sib.md" if rat_value == "nr" else "lte_sib.md"
    sib_file = Path(__file__).parent.parent / "data_sib" / sib_filename
    
    if not sib_file.exists():
        return []

    sib_lookup_table = sib_file.read_text(encoding="utf-8")
    target = normalise_system_info_combination(rat_value, combination)
    
    if not target:
        return []

    for raw_line in sib_lookup_table.splitlines():
        line = raw_line.strip()
        if not line.startswith("-") or ":" not in line:
            continue

        key_text, sequence_text = line[1:].split(":", 1)
        key = normalise_system_info_combination(rat_value, key_text)
        if key != target:
            continue

        return [
            {"name": msg.upper(), "direction": "GNB_TO_UE", "layer": "SYSTEM", "cell_id": ""}
            for msg in extract_sib_messages_from_sequence_text(sequence_text)
        ]

    return []
