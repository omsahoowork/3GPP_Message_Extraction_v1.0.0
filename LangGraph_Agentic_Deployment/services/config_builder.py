from __future__ import annotations

from typing import Any


def build_query_config(
    *,
    test_description: str,
    rat: str,
    additional_prompt: str,
) -> dict[str, Any]:
    rat_clean = str(rat or "").strip().upper()
    rat_value = "NR SA" if rat_clean == "NR" else "LTE"
    protocol_layers = "[RRC, NAS]" if rat_clean == "NR" else "[RRC, NAS]"

    config: dict[str, Any] = {
        "test_description": str(test_description or "").strip(),
        "RAT": rat_value,
        "Core": "5GC" if rat_clean == "NR" else "EPC",
        "Protocol Layers": protocol_layers,
        "Expected Outcome": "SUCCESS",
        "Cell Relation": "INTER CELL",
        "Frequency Relation": "INTER FREQUENCY",
    }

    extra = str(additional_prompt or "").strip()
    if extra:
        config["additional_user_query"] = extra

    return config
