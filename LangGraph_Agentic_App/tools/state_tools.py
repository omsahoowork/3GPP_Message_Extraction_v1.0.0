"""Tools for UE state transition and loop path resolution."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool


def normalise_state_key(state_value: str) -> str:
    """Normalize state key for lookup in state tables."""
    cleaned = re.sub(r"[\*_`]", "", str(state_value), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    cleaned = re.sub(r"\bSTATE\b", "", cleaned).strip()
    return cleaned


def load_state_loop_map(rat: str) -> dict[str, str]:
    """Load state-to-loop-path mapping from data_states/<rat>_states.md."""
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
        loop_path = cells[-1].replace("`", "").strip()
        if not state_cell or state_cell.lower() in {"state", "state id"}:
            continue
        if state_cell.startswith("---") or not loop_path:
            continue

        normalised = normalise_state_key(state_cell)
        if normalised:
            loop_map[normalised] = loop_path

        explicit = f"STATE {normalised}".strip()
        loop_map[explicit] = loop_path

    return loop_map


def extract_state_key_from_ue_state(ue_state: str, rat: str) -> str:
    """Extract state key from UE state text."""
    text = str(ue_state or "")
    if rat == "nr":
        match = re.search(r"\b\d[A-Z]-[A-Z]\b", text, flags=re.IGNORECASE)
        if match:
            return normalise_state_key(match.group(0))
    else:
        match = re.search(r"\bSTATE\s*([0-9][A-Z]?(?:-[A-Z0-9]+)?)\b", text, flags=re.IGNORECASE)
        if match:
            return normalise_state_key(match.group(1))

    return normalise_state_key(text)


@tool
def get_ue_state_loop_path(
    rat: Annotated[str, "RAT type: 'nr' or 'lte'"],
    ue_state: Annotated[str, "UE state text e.g. 'RRC_CONNECTED' or '1A'"],
) -> dict:
    """Get the loop path for a given UE state.

    Returns a dict with the loop path (list of states to traverse) and metadata.

    Args:
        rat: Either 'nr' or 'lte'
        ue_state: The UE state identifier

    Returns:
        Dict with keys: {loop_path: str|None, loop_states: list[str], found: bool}
    """
    loop_map = load_state_loop_map(rat)
    if not loop_map:
        return {"loop_path": None, "loop_states": [], "found": False}

    state_key = extract_state_key_from_ue_state(ue_state, rat)
    loop_path = loop_map.get(state_key) or loop_map.get(f"STATE {state_key}")
    
    if not loop_path:
        return {"loop_path": None, "loop_states": [], "found": False}

    # Parse loop path into list of states (e.g. "1A -> 1B -> 2A" becomes ["1A", "1B", "2A"])
    loop_states = [s.strip() for s in re.split(r"->|→", str(loop_path)) if s.strip()]
    
    return {
        "loop_path": loop_path,
        "loop_states": loop_states,
        "found": True,
    }


@tool
def retrieve_state_transition_context(
    rat: Annotated[str, "RAT type: 'nr' or 'lte'"],
    loop_path: Annotated[str, "Loop path string e.g. '1A -> 1B -> 2A'"],
) -> dict:
    """Retrieve transition/state context chunks for a given loop path.

    This is a wrapper that calls the retrieval tool from tools.retrieval_508_tool.invoke.

    Args:
        rat: Either 'nr' or 'lte'
        loop_path: The loop path to retrieve context for

    Returns:
        Dict with keys: {sequence_chunks: list|None, state_chunks: list|None,
                        combined_context: str, loop_path: list[str],
                        missing_transitions: list|None, missing_states: list|None}
    """
    from tools.retrieval_508_tool import retrieve_508_context
    
    try:
        result = retrieve_508_context.invoke({
            "rat": rat,
            "loop_path": loop_path,
        })
        return result
    except Exception as e:
        return {
            "sequence_chunks": None,
            "state_chunks": None,
            "combined_context": "",
            "loop_path": [],
            "missing_transitions": [str(e)],
            "missing_states": [],
            "error": str(e),
        }
