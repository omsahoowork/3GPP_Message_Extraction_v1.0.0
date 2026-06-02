"""SIB message extraction agent using ReAct pattern."""
from __future__ import annotations

from functools import lru_cache
import json
from typing import Annotated

from langgraph.prebuilt import create_react_agent

from core.llm import get_llm
from core.prompts import SIB_AGENT_SYSTEM_PROMPT
from tools.sib_tools import lookup_sib_combination, sib_lookup_table_text


@lru_cache(maxsize=8)
def create_sib_agent(llm_provider: str, llm_model: str):
    """Create a ReAct agent for SIB message extraction."""
    llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.1)
    
    tools = [lookup_sib_combination, sib_lookup_table_text]
    
    agent = create_react_agent(
        llm,
        tools,
        prompt=SIB_AGENT_SYSTEM_PROMPT,
    )
    return agent


def _coerce_agent_output_to_messages(raw_content: object, default_cell_id: str) -> list[dict]:
    """Coerce agent final content to a list of message dicts."""
    if isinstance(raw_content, list):
        if all(isinstance(item, dict) for item in raw_content):
            parsed = raw_content
        else:
            return []
    else:
        text = str(raw_content or "").strip()
        if not text:
            return []

        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))
            text = text.strip()

        try:
            parsed_json = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

        if isinstance(parsed_json, dict):
            candidate = parsed_json.get("messages") or parsed_json.get("message_sequence") or []
            parsed = candidate if isinstance(candidate, list) else []
        elif isinstance(parsed_json, list):
            parsed = parsed_json
        else:
            parsed = []

    normalised: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if not str(row.get("name", "")).strip():
            continue
        if not str(row.get("cell_id", "")).strip():
            row["cell_id"] = default_cell_id
        normalised.append(row)
    return normalised


def run_sib_agent(
    rat: str,
    combination: str,
    cell_id: str,
    llm_provider: str = "",
    llm_model: str = "",
    max_iterations: int = 10,
) -> list[dict]:
    """Run the SIB agent to extract SIB messages for a given combination.

    Args:
        rat: RAT type ('nr' or 'lte')
        combination: System information combination (e.g. 'NR-2' or 'system information combination 31')
        cell_id: Serving cell ID to assign to each SIB message
        max_iterations: Max reasoning steps before returning what we have

    Returns:
        List of SIB message dicts with {name, direction, cell_id, layer}
    """
    prompt = f"""\
Extract the SIB message sequence for:
- RAT: {rat}
- System information combination: {combination}
- Serving cell ID: {cell_id}

Use lookup_sib_combination first. If that returns empty, use sib_lookup_table_text to find the sequence.
Validate that the returned SIBs match the combination before finishing.
Return ONLY the list of message dicts, no explanation.
"""

    try:
        sib_agent = create_sib_agent(llm_provider, llm_model)
        # Invoke agent (returns {'output': <result>})
        result = sib_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": max_iterations},
        )

        # Parse the agent's final output.
        output = result.get("messages", [])
        if output:
            last = output[-1]
            content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
            coerced = _coerce_agent_output_to_messages(content, cell_id)
            if coerced:
                return coerced
        
        return []
    except Exception as e:
        print(f"SIB agent error: {e}")
        return []
